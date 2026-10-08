# 活塞智研 · 渤海活塞 × 深势科技

V2.0 交互概念演示。纯 HTML / CSS / JavaScript，无构建步骤、无后端、无 API Key、无外部脚本与字体依赖。

## GitHub Pages 发布

此目录只在独立分支 `piston-ai-demo` 中新增。主分支和原有研究代码未修改。

在仓库 **Settings → Pages → Build and deployment** 中设置：

- Source：**Deploy from a branch**
- Branch：**piston-ai-demo**
- Folder：**/docs**
- 点击 **Save**，等待 GitHub 完成部署后点击显示的访问地址。

这次源码提交不会自动启用 Pages；需要有管理权限的账号完成上述设置。`.nojekyll` 已包含，无须安装 Node.js 或配置构建命令。

GitHub 官方说明：https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## 演示场景

应用总览、材料知识问答、铝合金活塞铸造参数优化、钢活塞机加工参数优化、七类设备的21条样例记录、六份内置资料。可点击来源、查看原始样例、修改输入与目标、比较12/18组候选方案、生成验证计划并导出CSV和HTML报告。

顶部提供投屏模式、全屏入口、显示设置和一键演示。资料只在浏览器内查看或按关键词检索，不上传、不自动参与问答。手机布局可横向滑动导航栏与表格。

## 编辑与本地使用

- `index.html`：品牌区、导航、弹窗与页面骨架。
- `style.css`：蓝白视觉样式、投屏与响应式布局。
- `app.js`：样例数据、场景回复、交互和演示评分公式。

直接打开 `index.html`，或运行 `python3 -m http.server 8000 --directory docs` 后访问本地服务器。文件路径均为相对路径；`#home`、`#knowledge`、`#casting`、`#machining`、`#data`、`#docs` 可直接定位模块。

## 能力与品牌边界

全部试验记录、候选参数和性能结果均为人工构造，问答采用预置脚本与关键词匹配，优化采用启发式规则。没有连接真实模型、企业数据库、设备或生产系统。结果不能直接用于生产。无 PDF 解析、图片识别或新增文件自动问答。

渤海活塞盾形标识是参照公开样式制作的演示矢量版；深势科技采用文字标识排版，均非企业确认的官方 VI 母版。正式公开宣传前应替换为企业确认素材。标识不代表本系统已由企业官方上线或获得背书。
