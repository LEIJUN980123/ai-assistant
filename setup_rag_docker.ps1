# setup_rag_docker.ps1 (F盘专用版)
# 作者: Qwen
# 功能: 在 F 盘部署 RAG 项目 + 引导 Docker 安装（数据存 F 盘）

$TargetDrive = "F:"
$ProjectRoot = "$TargetDrive\RAG-App"

Write-Host "🚀 开始在 $TargetDrive 部署 RAG 项目..." -ForegroundColor Cyan

# 检查 F 盘是否存在
if (-not (Test-Path $TargetDrive)) {
    Write-Host "❌ $TargetDrive 盘不存在！请插入或创建 F 盘。" -ForegroundColor Red
    exit 1
}

# 创建项目目录（如果不存在）
if (-not (Test-Path $ProjectRoot)) {
    New-Item -ItemType Directory -Path $ProjectRoot -Force | Out-Null
    Write-Host "📁 项目目录已创建: $ProjectRoot" -ForegroundColor Green
}

# 切换到 F 盘项目目录
Set-Location $ProjectRoot

# === 后续逻辑：检查 Docker、安装、构建等（与原逻辑一致）===
# 注意：Dockerfile、data、.env 应放在 $ProjectRoot 下

Write-Host "`n🔍 步骤 1: 检查 Docker 是否已安装..." -ForegroundColor Yellow
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "✅ Docker 已安装" -ForegroundColor Green
} else {
    Write-Host "❌ Docker 未安装" -ForegroundColor Red
    $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $installerPath = "$env:TEMP\DockerDesktopInstaller.exe"

    Write-Host "📥 正在下载 Docker Desktop 安装程序..."
    try {
        Invoke-WebRequest -Uri $dockerUrl -OutFile $installerPath -TimeoutSec 120
    } catch {
        Write-Host "❌ 下载失败，请手动安装：" -ForegroundColor Red
        Write-Host "👉 https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
        exit 1
    }

    Write-Host "⚙️  启动 Docker Desktop 安装程序..." -ForegroundColor Yellow
    Write-Host "💡 安装后，请按以下步骤将 WSL2 数据移到 F 盘：" -ForegroundColor Magenta
    Write-Host "   1. 安装完成后重启电脑"
    Write-Host "   2. 运行 PowerShell（管理员）执行："
    Write-Host "      wsl --export Ubuntu ubuntu.tar"
    Write-Host "      wsl --unregister Ubuntu"
    Write-Host "      wsl --import Ubuntu F:\wsl\Ubuntu ubuntu.tar"
    Write-Host "   3. 设置默认用户（参考微软文档）"
    Start-Process -FilePath $installerPath -Wait
    Remove-Item $installerPath -Force
    Write-Host "✅ 请完成上述 WSL2 迁移后再继续！" -ForegroundColor Yellow
    pause
}

# === 检查 WSL2 ===
Write-Host "`n🔧 步骤 2: 检查 WSL2..." -ForegroundColor Yellow
try {
    $wslInfo = wsl -l -v 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ WSL2 已启用" -ForegroundColor Green
    } else {
        throw
    }
} catch {
    Write-Host "⚠️ 请先安装 WSL2 并迁移至 F 盘（见上文提示）" -ForegroundColor Red
    exit 1
}

# === 等待 Docker 就绪 ===
Write-Host "`n🔄 步骤 3: 等待 Docker 服务启动..." -ForegroundColor Yellow
$maxRetries = 30
$retryCount = 0
while ($retryCount -lt $maxRetries) {
    if ((docker version 2>$null) -match "Server") {
        break
    }
    Write-Host "." -NoNewline -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    $retryCount++
}
if ($retryCount -ge $maxRetries) {
    Write-Host "`n❌ Docker 未就绪，请手动启动 Docker Desktop！" -ForegroundColor Red
    exit 1
}
Write-Host "`n✅ Docker 已就绪！" -ForegroundColor Green

# === 构建和运行（从 F 盘）===
Write-Host "`n🏗️  步骤 4: 构建镜像（工作目录: $ProjectRoot）" -ForegroundColor Yellow
if (-not (Test-Path "Dockerfile")) {
    Write-Host "❌ 请将 Dockerfile 复制到 $ProjectRoot" -ForegroundColor Red
    exit 1
}

docker build -t rag-app:latest .

Write-Host "`n▶️  步骤 5: 启动容器（挂载 F 盘 data 目录）" -ForegroundColor Yellow
$hasEnv = Test-Path ".env"
$hasData = Test-Path "data\processed\Docker.json"

if ($hasEnv -and $hasData) {
    docker run --rm `
        --env-file .env `
        -v "${ProjectRoot}\data:/app/data:ro" `
        -p 7860:7860 `
        rag-app:latest
} else {
    Write-Host "⚠️  缺少 .env 或 data 文件，使用基础模式" -ForegroundColor Yellow
    docker run --rm -p 7860:7860 rag-app:latest
}

Write-Host "`n🎉 应用已从 $ProjectRoot 启动！访问 http://localhost:7860" -ForegroundColor Cyan