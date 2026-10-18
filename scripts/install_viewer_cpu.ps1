# EMU viewer-only Windows bootstrap. No CUDA, JAX, MuJoCo Python or training.
# Download this file via HTTPS and run it locally in Windows PowerShell.
# Installs missing Git, Node and Python through winget with normal consent/UAC.
[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $env:USERPROFILE 'emu-viewer-cpu'),
    [switch]$NoLaunch,
    [switch]$SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Branch = 'feature/v5017-goal-navigation-demo'
$Repository = 'https://github.com/mdt733/emu.git'
$ViewerUrl = 'http://127.0.0.1:8767/'

function Say([string]$Message) {
    Write-Host "[EMU viewer] $Message" -ForegroundColor Cyan
}

function Refresh-ToolsPath {
    # winget often edits the system/user PATH without updating this shell.
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @($env:Path, $machine, $user) + @(
        (Join-Path $env:ProgramFiles 'Git\cmd'),
        (Join-Path $env:ProgramFiles 'nodejs'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\Scripts'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps')
    )
    $env:Path = ($parts | Where-Object { $_ } | Select-Object -Unique) -join ';'
}

function Find-Executable([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandType -eq 'Application' } |
        Select-Object -First 1
    if ($null -ne $cmd) { return $cmd.Source }
    return $null
}

function Check-Node {
    $exe = Find-Executable 'node.exe'
    if (-not $exe) { return $false }
    try {
        $v = [version]((& $exe --version).Trim().TrimStart('v'))
        # Vite 7 requires Node >=22.12, or >=20.19; prefer supported LTS.
        return ($v.Major -gt 22 -or ($v.Major -eq 22 -and $v.Minor -ge 12))
    }
    catch { return $false }
}

function Python-Launcher {
    # Prefer a real Python launcher, not the Store's python.exe stub.
    $py = Find-Executable 'py.exe'
    if ($py) {
        foreach ($selector in @('-3.13', '-3')) {
            try {
                & $py $selector -c 'import sys;sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>$null
                if ($LASTEXITCODE -eq 0) {
                    return @{ Exe = $py; Prefix = @($selector) }
                }
            }
            catch {}
        }
    }
    foreach ($candidate in @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        (Join-Path $env:ProgramFiles 'Python313\python.exe'),
        (Find-Executable 'python.exe')
    )) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        try {
            & $candidate -c 'import sys;sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return @{ Exe = $candidate; Prefix = @() } }
        }
        catch {}
    }
    return $null
}

function Install-WithWinget([string]$Id, [string]$Label) {
    $winget = Find-Executable 'winget.exe'
    if (-not $winget) {
        throw "Missing $Label and winget. Install Microsoft's 'App Installer' (provides winget), then rerun this bootstrap."
    }
    Say "Installing $Label using Windows Package Manager (may request UAC approval)..."
    & $winget install --id $Id --exact --source winget --silent --accept-source-agreements --accept-package-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install $Label (exit code $LASTEXITCODE). Resolve the displayed installer error and rerun."
    }
    Refresh-ToolsPath
}

if ($env:OS -ne 'Windows_NT') {
    throw 'This bootstrap is specifically for Windows 10/11 PowerShell. For Linux/WSL use the manual viewer-only steps in docs/V5017_GOAL_NAVIGATION.md.'
}
if ($SelfTest) {
    # Safe CI smoke: parser and constants only. No downloads, package manager,
    # installation, subprocess server, or file changes.
    if ($Branch -ne 'feature/v5017-goal-navigation-demo' -or
        $ViewerUrl -ne 'http://127.0.0.1:8767/') {
        throw 'Bootstrap identity mismatch'
    }
    Say 'Bootstrap syntax, Windows compatibility and branch constants OK'
    exit 0
}

Refresh-ToolsPath

$git = Find-Executable 'git.exe'
if (-not $git) {
    Install-WithWinget -Id 'Git.Git' -Label 'Git for Windows'
    $git = Find-Executable 'git.exe'
}
if (-not $git) { throw 'Git was installed but not detected; reopen PowerShell and rerun the installer.' }
Say "Git: $(& $git --version)"

$npm = Find-Executable 'npm.cmd'
if (-not (Check-Node) -or -not $npm) {
    Install-WithWinget -Id 'OpenJS.NodeJS.LTS' -Label 'Node.js LTS and npm'
    $npm = Find-Executable 'npm.cmd'
}
if (-not (Check-Node) -or -not $npm) {
    throw 'Node.js/npm are missing or too old after installation; reopen PowerShell and rerun the installer.'
}
$node = Find-Executable 'node.exe'
Say "Node: $(& $node --version); npm: $(& $npm --version)"

$python = Python-Launcher
if (-not $python) {
    Install-WithWinget -Id 'Python.Python.3.13' -Label 'Python 3.13 (stdlib only)'
    $python = Python-Launcher
}
if (-not $python) {
    throw 'Python 3.10+ was not detected after installation; reopen PowerShell and rerun the installer.'
}
Say "Python: $(& $python.Exe @($python.Prefix) --version)"

$Destination = [IO.Path]::GetFullPath($Destination)
$gitFolder = Join-Path $Destination '.git'
if (Test-Path -LiteralPath $gitFolder -PathType Container) {
    Say "Existing viewer found: $Destination"
    $currentBranch = (& $git -C $Destination branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $currentBranch -ne $Branch) {
        throw "Refusing to update unexpected branch '$currentBranch'. Expected '$Branch' in $Destination."
    }
    $changes = & $git -C $Destination status --porcelain
    if ($LASTEXITCODE -ne 0 -or $changes) {
        throw "Checkout has local edits. Preserving them; commit/stash changes before rerunning: $Destination"
    }
    Say 'Updating viewer branch (fast-forward only)...'
    & $git -C $Destination pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Viewer update failed; no files were forcibly reset.' }
} elseif (Test-Path -LiteralPath $Destination) {
    if (@(Get-ChildItem -LiteralPath $Destination -Force).Count -gt 0) {
        throw "Destination is not an EMU checkout and is not empty: $Destination"
    }
    Say "Cloning experimental CPU viewer into $Destination..."
    & $git clone --single-branch --branch $Branch $Repository $Destination
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
} else {
    Say "Cloning experimental CPU viewer into $Destination..."
    & $git clone --single-branch --branch $Branch $Repository $Destination
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
}

$viewer = Join-Path $Destination 'emu\viewer_src'
$prepare = Join-Path $Destination 'scripts\prepare_showcase.py'
$champion = Join-Path $Destination 'showcase\champion.emupolicy'
$lock = Join-Path $viewer 'package-lock.json'
foreach ($required in @($prepare, $champion, $lock)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Incomplete repository checkout: missing $required"
    }
}

Say 'Validating and bundling the trained 260M champion (no Python packages)...'
Push-Location $Destination
try {
    & $python.Exe @($python.Prefix) $prepare --require-champion
    if ($LASTEXITCODE -ne 0) { throw 'Champion asset preparation failed.' }

    Say 'Installing locked browser dependencies only (no training libraries)...'
    & $npm ci --prefix $viewer --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }

    Say 'Checking browser navigation and control contracts...'
    & $npm test --prefix $viewer
    if ($LASTEXITCODE -ne 0) { throw 'Browser tests failed; viewer was not started.' }
} finally { Pop-Location }

Say "CPU viewer ready: $ViewerUrl"
Say 'Choose Settings -> Demo controls -> Navigate random poses, then select a route planner.'
Say 'No CUDA, JAX, ML training environment or large PKL was installed.'
if ($NoLaunch) { return }

# Run the dev server in this same terminal. Ctrl+C stops it cleanly.
# Avoid opening a browser before localhost actually starts (error page).
Say "Starting viewer on $ViewerUrl (leave this terminal open; Ctrl+C stops it)..."
Push-Location $Destination
try {
    & $npm run dev --prefix $viewer -- --host 127.0.0.1 --port 8767 --strictPort
    if ($LASTEXITCODE -ne 0) { throw 'Viewer server exited with an error.' }
} finally { Pop-Location }
