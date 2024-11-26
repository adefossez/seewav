import os
import subprocess
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def predict(
        self,
        audio: Path = Input(description="Audio file (mp3 or wav)."),
        rate: int = Input(description="Video framerate.", default=60),
        stereo: bool = Input(description="Create 2 waveforms for stereo files.", default=False),
        color: str = Input(description="Color of the bars as 'r,g,b' in [0,1].", default="0.03,0.6,0.3"),
        color2: str = Input(description="Color of the second waveform for stereo files.", default="0.5,0.3,0.6"),
        white: bool = Input(description="Use white background.", default=False),
        bars: int = Input(description="Number of bars on the video at once.", default=50),
        oversample: float = Input(description="Lower values will feel less reactive.", default=4.0),
        time_param: float = Input(description="Amount of audio shown at once on a frame.", default=0.4),
        speed: float = Input(description="Faster transitions between frames.", default=4.0),
        width: int = Input(description="Width of the animation in pixels.", default=480),
        height: int = Input(description="Height of the animation in pixels.", default=300),
        seek: float = Input(description="Seek to time in seconds in video.", default=None),
        duration: float = Input(description="Duration in seconds from seek time.", default=None),
    ) -> Path:
        """
        Procesa el archivo de audio subido y devuelve un archivo MP4.
        """
        # Ruta de salida para el archivo generado
        output_path = Path("/src/out.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Construir el comando como una lista
        command = [
            "/root/.pyenv/shims/python3",
            os.path.join('/src/seewav.py'),
            "-r", str(rate),
            "-c", color,
            "-c2", color2,
            "-B", str(bars),
            "-O", str(oversample),
            "-T", str(time_param),
            "-S", str(speed),
            "-W", str(width),
            "-H", str(height),
            str(audio),
            str(output_path)
        ]

        if stereo:
            command.append("--stereo")
        if white:
            command.append("--white")
        if seek is not None:
            command.extend(["-s", str(seek)])
        if duration is not None:
            command.extend(["-d", str(duration)])

        # Imprimir el comando para depuración
        print("Comando a ejecutar:", " ".join(command))

        # Ejecutar el comando y capturar la salida
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Imprimir la salida y errores para depuración
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        # Verificar si el archivo de salida se ha generado correctamente
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("El archivo de salida no se ha generado correctamente.")

        return output_path  # Devolver la ruta del archivo de salida
