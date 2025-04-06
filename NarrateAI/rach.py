# def build_srt(entries, output_file):
#     with open(output_file, "w", encoding="utf-8") as f:
#         for i, (start, end, description) in enumerate(entries, start=1):
#             start_timecode = format_timecode(start)
#             end_timecode = format_timecode(end)
#             f.write(f"{i}\n{start_timecode} --> {end_timecode}\n{description}\n\n")

# def format_timecode(seconds):
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     seconds = int(seconds % 60)
#     milliseconds = int((seconds % 1) * 1000)
#     return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}" 

def build_srt(entries, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, description) in enumerate(entries, start=1):
            start_timecode = format_timecode(start)
            end_timecode = format_timecode(end)
            f.write(f"{i}\n{start_timecode} --> {end_timecode}\n{description}\n\n")

def format_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
