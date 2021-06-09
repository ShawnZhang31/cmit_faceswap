#!/bin/sh

# 日志文件
touch /cmit_faceswap/logs/access.log
touch /cmit_faceswap/logs/error.log
# touch /cmit_faceswap/log/output.log

exec gunicorn -b 0.0.0.0:5000 -w 4 cmit_faceswap:app --access-logfile=/cmit_faceswap/logs/access.log --error-logfile=/cmit_faceswap/logs/error.log --timeout=180