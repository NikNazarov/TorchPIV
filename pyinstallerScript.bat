if exist ".\lightTorchVenv" (
    call .\lightTorchVenv\Scripts\activate.bat
) else if exist ".\venv" (
    call .\venv\Scripts\activate.bat
)
else (
    call activate base
)
pyinstaller --noconsole --onedir main.py ^
--copy-metadata imageio ^
--hidden-import imageio_ffmpeg ^
--hidden-import imageio.plugins.ffmpeg ^
--collect-submodules imageio ^
--icon TorchPIVLogo.png ^
--name TorchPIV ^
--splash TorchPIVLogo.png ^
--noconfirm 
call deactivate