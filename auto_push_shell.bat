@echo off
prompt $G
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置远程仓库地址和分支
set BRANCH=main
set GITHUB_URL=https://github.com/Zhangxulang/Deep-Learning.git

:: 确保在当前仓库路径下执行（可选）

:: 拉取最新代码，避免冲突
echo 正在从远程分支 %BRANCH% 拉取最新更改...
git pull origin %BRANCH%

:: 生成当前时间字符串作为提交备注
set TIME_STR=%date:~0,10%_%time:~0,5%
set TIME_STR=%TIME_STR: =_%
set TIME_STR=%TIME_STR::=-%
set TIME_STR=%TIME_STR:,=-%

:: 自动添加所有变更并提交
echo 正在提交本地更改：%TIME_STR%
git add .
git commit -m "自动提交更新 - %TIME_STR%" >nul 2>nul

:: 检查是否已设置 origin，如未设置则添加
git remote get-url origin >nul 2>nul
if errorlevel 1 (
    echo 正在添加远程仓库 origin...
    git remote add origin %GITHUB_URL%
)

:: 推送到 GitHub 仓库
echo 正在推送到 GitHub 仓库...
git push origin %BRANCH%

:: 打开仓库网页
start https://github.com/Zhangxulang/Deep-Learning

echo.
echo 所有操作完成！
pause