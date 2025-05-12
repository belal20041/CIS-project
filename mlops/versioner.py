import subprocess
class DVCVersioner:
    def add_file(self, filepath):
        subprocess.run(["dvc", "add", filepath])
