# no-lose-janken-ai
コマンドの順番  
1.\ docker build -t janken-app .
2.\ docker run -it --rm --device=/dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    janken-app

