# Purpose: turn a start-time-in-seconds float (e.g. TrimToExactDuration.actual_start_seconds)
# into a zero-padded, sortable timecode and bake it into a VHS_VideoCombine filename_prefix.

class TimecodeFilenamePrefix:
    """
    Builds a filename prefix that encodes the song position of the clip.
    Output example with base 'clip', seconds 12.0, fps 25:
        clip_00-12-00
    so in Resolve you scrub straight to 00:12:00.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prefix": ("STRING", {"default": "clip", "multiline": False}),
                "seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.001}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "separator": ("STRING", {"default": "_", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("filename_prefix", "timecode", "frame")
    FUNCTION = "build"
    CATEGORY = "kubakub/utils"

    def build(self, base_prefix, seconds, fps, separator):
        fps_i = max(1, int(round(fps)))

        # frame-accurate decomposition: total frames first, then split.
        total_frames = int(round(seconds * fps))
        ff = total_frames % fps_i
        total_secs = total_frames // fps_i
        ss = total_secs % 60
        mm = total_secs // 60

        timecode = f"{mm:02d}-{ss:02d}-{ff:02d}"          # e.g. 00-12-00
        prefix = f"{base_prefix}{separator}{timecode}"     # e.g. clip_00-12-00

        return (prefix, timecode, total_frames)


NODE_CLASS_MAPPINGS = {
    "TimecodeFilenamePrefix": TimecodeFilenamePrefix,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TimecodeFilenamePrefix": "kubakub timecode filename prefix",
}
