"""
FFProbe Analyzer Module
Provides frame-level analysis of M2TS files for BDInfo-style chapter bitrate statistics
"""

import subprocess
import json
import os
import logging
from collections import deque


# Constants for default values
DEFAULT_BITRATE = "0 kbps"
DEFAULT_SIZE = "0 bytes"
DEFAULT_TIMESTAMP = "00:00:00.000"

# Configure logging
logger = logging.getLogger(__name__)


class FrameAnalysis:
    """Analyzes frame-level data from M2TS files using ffprobe"""

    def __init__(self):
        """Initialize frame analyzer with cache"""
        self.frame_cache: dict[str, list[dict]] = {}

    def clear_cache(self) -> None:
        """Clear the frame data cache"""
        self.frame_cache.clear()

    def get_frame_data(self, m2ts_file: str, start_time: float = 0.0,
                      end_time: float | None = None) -> list[dict]:
        """
        Get frame-by-frame data from M2TS file using ffprobe

        Args:
            m2ts_file: Path to M2TS file
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: None for full file)

        Returns:
            list of frame data dictionaries
        """
        cache_key = f"{m2ts_file}_{start_time}_{end_time}"
        if cache_key in self.frame_cache:
            logger.debug(f"Using cached frame data for {cache_key} ({len(self.frame_cache[cache_key])} frames)")
            return self.frame_cache[cache_key]

        try:
            if not os.path.exists(m2ts_file):
                logger.warning(f"M2TS file not found: {m2ts_file}")
                return []

            file_size = os.path.getsize(m2ts_file)
            analysis_duration = None if end_time is None else end_time - start_time

            logger.info(f"Analyzing {m2ts_file} (size: {file_size / (1024**3):.1f} GB)")

            timeout_seconds = 180  # 3 minutes max for analysis

            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_frames',
                '-select_streams', 'v:0',
                '-print_format', 'json'
            ]

            # Add time range parameters
            if start_time > 0 and analysis_duration is not None:
                interval_end = start_time + analysis_duration
                cmd.extend(['-read_intervals', f"{start_time:.3f}%{interval_end:.3f}"])
            elif start_time > 0:
                cmd.extend(['-read_intervals', f"{start_time:.3f}%+"])
            elif analysis_duration is not None:
                cmd.extend(['-read_intervals', f"%{analysis_duration:.3f}"])

            cmd.append(m2ts_file)

            logger.debug(f"Running ffprobe command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)

            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {m2ts_file} (return code: {result.returncode})")
                if result.stderr:
                    logger.debug(f"ffprobe stderr: {result.stderr}")
                return []

            if not result.stdout.strip():
                logger.warning(f"ffprobe returned empty output for {m2ts_file}")
                return []

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse ffprobe JSON output for {m2ts_file}: {e}")
                return []

            frames = data.get('frames', [])
            video_frames = []

            for frame in frames:
                if frame.get('media_type') == 'video':
                    raw_pts_time = float(frame.get('pkt_pts_time', 0))
                    adjusted_pts_time = raw_pts_time + start_time

                    frame_data = {
                        'pts_time': adjusted_pts_time,
                        'duration_time': float(frame.get('pkt_duration_time', 0)),
                        'size': int(frame.get('pkt_size', 0)),
                        'key_frame': frame.get('key_frame', 0) == 1,
                        'pict_type': frame.get('pict_type', ''),
                        'width': int(frame.get('width', 0)),
                        'height': int(frame.get('height', 0))
                    }
                    video_frames.append(frame_data)

            logger.info(f"Analyzed {len(video_frames)} video frames from {m2ts_file}")
            self.frame_cache[cache_key] = video_frames
            return video_frames

        except subprocess.TimeoutExpired:
            logger.warning(f"ffprobe timed out after 180s while analyzing {m2ts_file}")
            self.frame_cache[cache_key] = []
            return []
        except Exception as e:
            logger.error(f"Unexpected error while analyzing frames for {m2ts_file}: {e}")
            self.frame_cache[cache_key] = []
            return []

    def analyze_chapter_bitrates(self, frames: list[dict], chapter_start: float,
                               chapter_end: float, video_bitrate: int = 0) -> dict[str, str]:
        """
        Analyze bitrate statistics for a chapter using frame data

        Args:
            frames: list of frame data
            chapter_start: Chapter start time in seconds
            chapter_end: Chapter end time in seconds
            video_bitrate: Overall video bitrate (fallback)

        Returns:
            Chapter bitrate statistics
        """
        chapter_frames = []
        max_sampled_time = 0

        for frame in frames:
            frame_time = frame['pts_time']
            if chapter_start <= frame_time < chapter_end:
                chapter_frames.append(frame)
            if frame_time > max_sampled_time:
                max_sampled_time = frame_time

        chapter_frames.sort(key=lambda x: x['pts_time'])

        # Check if chapter extends beyond sampled data
        if chapter_end > max_sampled_time and max_sampled_time > 0 and not chapter_frames:
            return self._get_fallback_stats(video_bitrate)

        if not chapter_frames:
            return self._get_fallback_stats(video_bitrate)

        # Calculate basic statistics
        total_bits = sum(frame['size'] * 8 for frame in chapter_frames)
        chapter_duration = chapter_end - chapter_start

        # For very short chapters, use frame duration sum
        if chapter_duration < 0.1:
            frame_duration_sum = sum(frame.get('duration_time', 0.04) for frame in chapter_frames)
            if frame_duration_sum > 0:
                chapter_duration = frame_duration_sum

        avg_bitrate = total_bits / chapter_duration if chapter_duration > 0 else 0

        # Frame size statistics
        frame_sizes = [frame['size'] for frame in chapter_frames]
        avg_frame_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
        max_frame_size = max(frame_sizes) if frame_sizes else 0
        max_frame_time = chapter_frames[frame_sizes.index(max_frame_size)]['pts_time'] if frame_sizes else 0

        # Sliding window analysis for peak bitrates
        actual_chapter_duration = chapter_end - chapter_start
        window1_peak, window1_time = self._calculate_peak_bitrate(
            chapter_frames, chapter_start, chapter_end, min(1.0, actual_chapter_duration))
        window5_peak, window5_time = self._calculate_peak_bitrate(
            chapter_frames, chapter_start, chapter_end, min(5.0, actual_chapter_duration))
        window10_peak, window10_time = self._calculate_peak_bitrate(
            chapter_frames, chapter_start, chapter_end, min(10.0, actual_chapter_duration))

        return {
            'avg_video_rate': f"{int(avg_bitrate / 1000)} kbps",
            'max_1sec_rate': f"{int(window1_peak / 1000)} kbps",
            'max_1sec_time': self._format_timestamp(window1_time),
            'max_5sec_rate': f"{int(window5_peak / 1000)} kbps",
            'max_5sec_time': self._format_timestamp(window5_time),
            'max_10sec_rate': f"{int(window10_peak / 1000)} kbps",
            'max_10sec_time': self._format_timestamp(window10_time),
            'avg_frame_size': f"{int(avg_frame_size)} bytes",
            'max_frame_size': f"{int(max_frame_size)} bytes",
            'max_frame_time': self._format_timestamp(max_frame_time)
        }

    def _get_fallback_stats(self, video_bitrate: int) -> dict[str, str]:
        """Get fallback statistics when no frame data is available"""
        fallback_rate = f"{video_bitrate // 1000} kbps" if video_bitrate > 0 else DEFAULT_BITRATE
        return {
            'avg_video_rate': fallback_rate,
            'max_1sec_rate': fallback_rate,
            'max_1sec_time': DEFAULT_TIMESTAMP,
            'max_5sec_rate': fallback_rate,
            'max_5sec_time': DEFAULT_TIMESTAMP,
            'max_10sec_rate': fallback_rate,
            'max_10sec_time': DEFAULT_TIMESTAMP,
            'avg_frame_size': DEFAULT_SIZE,
            'max_frame_size': DEFAULT_SIZE,
            'max_frame_time': DEFAULT_TIMESTAMP
        }

    def _calculate_peak_bitrate(self, frames: list[dict], chapter_start: float,
                              chapter_end: float, window_size: float) -> tuple[float, float]:
        """
        Calculate peak bitrate over a sliding window

        Args:
            frames: list of frame data
            chapter_start: Chapter start time
            chapter_end: Chapter end time
            window_size: Window size in seconds

        Returns:
            Tuple of (peak_bitrate, peak_time)
        """
        if not frames:
            return 0.0, 0.0

        chapter_frames = [f for f in frames if chapter_start <= f['pts_time'] < chapter_end]
        if not chapter_frames:
            return 0.0, 0.0

        actual_chapter_duration = chapter_end - chapter_start

        if actual_chapter_duration <= window_size:
            return self._calculate_single_window_bitrate(chapter_frames, chapter_start, actual_chapter_duration)
        else:
            return self._calculate_sliding_window_bitrate(chapter_frames, window_size)

    def _calculate_single_window_bitrate(self, frames: list[dict],
                                       chapter_start: float, duration: float) -> tuple[float, float]:
        """Calculate bitrate for chapters shorter than window size"""
        total_bits = sum(frame['size'] * 8 for frame in frames)

        if duration > 0:
            return total_bits / duration, chapter_start
        else:
            frame_duration_sum = sum(frame.get('duration_time', 0.04) for frame in frames)
            if frame_duration_sum > 0:
                return total_bits / frame_duration_sum, chapter_start
            else:
                return 0.0, chapter_start

    def _calculate_sliding_window_bitrate(self, frames: list[dict], window_size: float) -> tuple[float, float]:
        """Calculate peak bitrate using sliding window approach"""
        window_bits = deque()
        window_times = deque()
        window_bits_sum = 0.0
        peak_bitrate = 0.0
        peak_time = 0.0

        for frame in frames:
            frame_time = frame['pts_time']
            frame_bits = frame['size'] * 8

            window_bits.append(frame_bits)
            window_times.append(frame_time)
            window_bits_sum += frame_bits

            # Calculate window duration
            if len(window_times) > 1:
                window_times_sum = window_times[-1] - window_times[0]
            else:
                window_times_sum = 0

            # Check if window is filled or at end of frames
            if window_times_sum >= window_size or frame == frames[-1]:
                if window_times_sum > 0:
                    window_bitrate = window_bits_sum / window_times_sum
                else:
                    frame_duration = frame.get('duration_time', 0.04) or 0.04
                    window_bitrate = frame_bits / frame_duration

                if window_bitrate > peak_bitrate:
                    peak_bitrate = window_bitrate
                    peak_time = window_times[0] if window_times else frame_time

                # Remove oldest frame to maintain window size
                if window_times_sum >= window_size:
                    window_bits_sum -= window_bits.popleft()
                    window_times.popleft()

        return peak_bitrate, peak_time

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds to h:m:s.ms format

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        if seconds <= 0:
            return DEFAULT_TIMESTAMP

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs - int(secs)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{milliseconds:03d}"

    def analyze_m2ts_file(self, m2ts_file: str, video_bitrate: int = 0) -> dict:
        """
        Analyze an M2TS file and return frame statistics

        Args:
            m2ts_file: Path to M2TS file
            video_bitrate: Overall video bitrate for fallback

        Returns:
            Analysis results
        """
        frames = self.get_frame_data(m2ts_file)

        if not frames:
            return {
                'frames': [],
                'total_frames': 0,
                'total_duration': 0,
                'avg_bitrate': video_bitrate
            }

        total_duration = frames[-1]['pts_time'] + frames[-1]['duration_time'] if frames else 0
        total_bits = sum(frame['size'] * 8 for frame in frames)
        avg_bitrate = total_bits / total_duration if total_duration > 0 else video_bitrate

        return {
            'frames': frames,
            'total_frames': len(frames),
            'total_duration': total_duration,
            'avg_bitrate': avg_bitrate
        }


# Global instance for convenience
frame_analyzer = FrameAnalysis()
