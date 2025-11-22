#!/bin/bash
# GitHub推送脚本

cd ~/Desktop/Multi-Objective-Optimization-of-Piston-Steel-Composition-Using-Attention-Based-Neural-Networks

echo "正在尝试推送到GitHub..."
echo ""
echo "如果提示输入用户名，请输入: ustbTobyMa"
echo "如果提示输入密码，请输入您的GitHub Personal Access Token"
echo ""
echo "如果没有Token，请访问: https://github.com/settings/tokens"
echo "创建新的token (classic)，勾选 'repo' 权限"
echo ""

git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推送成功！"
else
    echo ""
    echo "❌ 推送失败，请检查："
    echo "1. 是否已创建Personal Access Token"
    echo "2. Token是否有 'repo' 权限"
    echo "3. 网络连接是否正常"
fi
