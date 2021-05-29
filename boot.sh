#!/bin/sh

# 日志文件
touch /cmit_faceswap/log/access.log
touch /cmit_faceswap/log/error.log
# touch /cmit_faceswap/log/output.log

exec gunicorn -b 0.0.0.0:5000 -w 2 cmit_faceswap:app --access-logfile=/cmit_faceswap/log/access.log --error-logfile=/cmit_faceswap/log/error.log --timeout=180 --log-level=debug --preload