# install_chinese_tessdata.ps1
# 自动下载简体/繁体中文语言包到 F:\Tesseract-OCR\tessdata

$ErrorActionPreference = "Stop"
$TessDataDir = "F:\Tesseract-OCR\tessdata"
$Languages = @("chi_sim", "chi_tra")
$ProxyBaseUrl = "https://ghproxy.com/https://github.com/tesseract-ocr/tessdata/raw/main/"

# 创建 tessdata 目录
if (!(Test-Path $TessDataDir)) {
    Write-Host "📁 创建目录: $TessDataDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $TessDataDir -Force | Out-Null
}

# 下载每个语言包
foreach ($lang in $Languages) {
    $LocalFile = Join-Path $TessDataDir "$lang.traineddata"
    if (Test-Path $LocalFile) {
        Write-Host "ℹ️  $lang 已存在，跳过下载" -ForegroundColor Gray
        continue
    }

    $Url = $ProxyBaseUrl + "$lang.traineddata"
    Write-Host "📥 正在下载 $lang.traineddata ..." -ForegroundColor Yellow

    try {
        Invoke-WebRequest -Uri $Url -OutFile $LocalFile -UseBasicParsing -TimeoutSec 60
        Write-Host "✅ $lang 下载成功" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ 下载失败: $lang | 错误: $_" -ForegroundColor Red
        Write-Host "💡 建议手动下载并放入: $LocalFile" -ForegroundColor Yellow
    }
}

# 验证：列出所有可用语言
Write-Host "`n🔍 验证 Tesseract 支持的语言..." -ForegroundColor White
try {
    $output = & "F:\Tesseract-OCR\tesseract.exe" --list-langs 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "📄 当前支持的语言："
        Write-Host $output
        if ($output -like "*chi_sim*" -and $output -like "*chi_tra*") {
            Write-Host "`n🎉 简体(chi_sim)和繁体(chi_tra)已成功启用！" -ForegroundColor Green
        } else {
            Write-Host "`n⚠️  中文语言包未生效，请检查文件是否在正确目录。" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ 无法调用 tesseract.exe，请确认已配置 PATH 或重启终端。" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ 验证失败: $_" -ForegroundColor Red
}

Write-Host "`n📌 使用方式（Python）：" -ForegroundColor Cyan
Write-Host 'pytesseract.image_to_string(img, lang="chi_sim+eng")   # 简体'
Write-Host 'pytesseract.image_to_string(img, lang="chi_tra+eng")   # 繁体'