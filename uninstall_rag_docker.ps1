# uninstall_rag_docker.ps1
# 作者: Qwen
# 功能: 卸载 Docker Desktop + WSL2 Ubuntu + 清理残留

Write-Host "🗑️ 开始卸载 RAG Docker 环境..." -ForegroundColor Red

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "❌ 请以管理员身份运行此卸载脚本！" -ForegroundColor Red
    exit 1
}

# 1. 停止并移除正在运行的容器（如果存在）
Write-Host "⏹️  停止并移除 RAG 容器..."
docker stop rag-app 2>$null
docker rm rag-app 2>$null

# 2. 删除镜像
Write-Host "🧹 删除 RAG 镜像..."
docker rmi rag-app:latest 2>$null

# 3. 卸载 Docker Desktop
Write-Host "🚮 卸载 Docker Desktop..."
$app = Get-WmiObject -Class Win32_Product | Where-Object { $_.Name -like "*Docker*" }
if ($app) {
    foreach ($a in $app) {
        Write-Host "正在卸载 $($a.Name)..."
        $a.Uninstall() | Out-Null
    }
} else {
    Write-Host "⚠️ Docker Desktop 未通过 MSI 安装，尝试手动卸载..."
    # 尝试调用官方卸载程序（如果存在）
    if (Test-Path "$env:ProgramFiles\Docker\Docker\Uninstall.exe") {
        & "$env:ProgramFiles\Docker\Docker\Uninstall.exe" /S
    }
}

# 4. 卸载 WSL2 Ubuntu 发行版（谨慎！会删除所有 Ubuntu 数据）
Write-Host "🌍 卸载 WSL2 Ubuntu 发行版..."
wsl --unregister Ubuntu 2>$null

# 5. 可选：禁用 WSL 功能（不推荐，除非你确定不需要）
# dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
# dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /norestart

# 6. 清理临时文件
Remove-Item "$env:TEMP\DockerDesktopInstaller.exe" -Force -ErrorAction SilentlyContinue

Write-Host "`n✅ 卸载完成！建议重启电脑以彻底清理内核组件。" -ForegroundColor Green
Write-Host "💡 如需重新安装，请运行 setup_rag_docker.ps1（修改版）" -ForegroundColor Cyan