import os
import tempfile


def ensure_project_cwd(project_root: str) -> None:
    os.chdir(project_root)


def save_upload_to_temp(uploaded_file, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
