#!/bin/bash
# 快速推送脚本 - 使用GitHub CLI或HTTPS

cd "$(dirname "$0")"

echo "=========================================="
echo "推送到GitHub"
echo "=========================================="
echo ""

# 方法1: 尝试使用GitHub CLI
if command -v gh &> /dev/null; then
    echo "检测到GitHub CLI，尝试使用..."
    gh auth status 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "GitHub CLI已认证，使用CLI推送..."
        git push origin main
        exit $?
    else
        echo "GitHub CLI未认证，请运行: gh auth login"
    fi
fi

# 方法2: 使用HTTPS（需要token）
echo ""
echo "使用HTTPS方式推送..."
echo "提示：如果要求输入密码，请使用GitHub Personal Access Token"
echo "创建Token: https://github.com/settings/tokens"
echo ""
git push origin main

