from pathlib import Path


def update_frame_labels(folder: str) -> None:
	label_dir = Path(folder)
	if not label_dir.is_dir():
		raise FileNotFoundError(f"Folder not found: {label_dir}")

	changed_files = 0
	for file_path in sorted(label_dir.glob("frame_*.txt")):
		text = file_path.read_text(encoding="utf-8").splitlines()
		updated_lines = []
		changed = False

		for line in text:
			parts = line.split()
			if not parts:
				updated_lines.append(line)
				continue

			if parts[0] == "17":
				parts[0] = "0"
				changed = True
			elif parts[0] == "16":
				parts[0] = "1"
				changed = True

			updated_lines.append(" ".join(parts))

		if changed:
			file_path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
			changed_files += 1

	print(f"Updated {changed_files} file(s) in {label_dir}")


if __name__ == "__main__":
	update_frame_labels("/home/abuntu/Documents/github_Temp/unknown-ship-detection-ai/temp/labels/long")
