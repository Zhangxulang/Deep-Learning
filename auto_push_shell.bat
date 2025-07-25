@echo off
prompt $G
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置远程仓库地址和分支
set GITHUB_URL=https://github.com/Zhangxulang/Deep-Learning.git
set BRANCH=main

:: 自动拉取，避免冲突
echo Pulling latest changes from remote branch %BRANCH%...
git pull origin %BRANCH%

:: 自动添加所有更改并提交
set TIME_STR=%date:~0,10%_%time:~0,5%
set TIME_STR=%TIME_STR: =_%
set TIME_STR=%TIME_STR::=-%
echo 自动提交中：%TIME_STR%...
git add .
git commit -m "自动提交更新 - %TIME_STR%" >nul 2>nul

:: 清空 origin，防止重复 push 地址
git remote remove origin >nul 2>nul

:: 重新设置 fetch（origin）
git remote add origin %GITEE_URL%

:: 添加双 push 地址
git remote set-url --add --push origin %GITHUB_URL%

:: 显示当前远程信息
echo.
echo 当前远程仓库配置：
git remote -v

:: 推送
echo.
echo 正在推送到  GitHub...
git push origin %BRANCH%

:: 自动打开网页查看
start https://github.com/Zhangxulang/Deep-Learning

echo.
echo 所有操作完成！
pause