# Laptop-safety watchdog (copied from GeoAD): kills any running ExplainAD job (python pipeline.py ...) the moment a
# limit is crossed. Runs alongside every heavy stage; every job here is resumable, so a
# kill costs minutes. While a job or dataset download runs it also keeps Windows from
# sleeping (sleep killed a download's connections on 09-27); the request is dropped as
# soon as nothing runs or this script exits. The screen may still turn off.
# Log: results/watchdog.log
#   powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1
param(
    [int]$MaxGpuTempC   = 83,    # RTX 4060 Laptop throttles ~87 C; stop well before
    [int]$MaxVramMiB    = 7000,  # of 8188; leave room for the desktop
    [int]$MaxPowerW     = 115,   # laptop 4060 TGP ceiling is ~115-140 W
    [int]$MinFreeRamMiB = 1500,
    [int]$MinFreeDiskGB = 20,
    [int]$IntervalSec   = 3,
    [switch]$Once              # single check, for testing
)
$log = Join-Path $PSScriptRoot "..\results\watchdog.log"

function Log($m) { $l = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"; Add-Content $log $l; Write-Output $l }

function Jobs {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'pipeline.py|improve.py|explainad.' }
    # dataset unpacks: no GPU, but they can fill the disk and must not be slept through
    Get-CimInstance Win32_Process -Filter "Name='tar.exe'" |
        Where-Object { $_.CommandLine -match 'mvtec' }
}

function Downloads {
    Get-CimInstance Win32_Process -Filter "Name='curl.exe'" |
        Where-Object { $_.CommandLine -match 'mydrive\.ch|amazon-visual-anomaly|drive\.usercontent' }
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
$awake = $false
$strikes = 0

Log "watchdog start: temp<$MaxGpuTempC C vram<$MaxVramMiB MiB power<$MaxPowerW W ram>$MinFreeRamMiB MiB disk>$MinFreeDiskGB GB"
while ($true) {
    $problems = @()
    $g = (& nvidia-smi --query-gpu=temperature.gpu,memory.used,power.draw --format=csv,noheader,nounits) -split ',\s*'
    if ($LASTEXITCODE -ne 0 -or $g.Count -lt 3) {
        $problems += "nvidia-smi unreadable"
    } else {
        $temp = [double]$g[0]; $vram = [double]$g[1]; $pw = [double]$g[2]
        if ($temp -ge $MaxGpuTempC) { $problems += "GPU temp $temp C" }
        if ($vram -ge $MaxVramMiB)  { $problems += "VRAM $vram MiB" }
        # nvidia-smi on this laptop intermittently returns exactly 590.01 W while idle at
        # ~12 W, sometimes on 4+ consecutive checks (09-27 21:11). A laptop 4060 cannot
        # draw >~140 W, so anything above 300 is a glitch; temperature still guards heat.
        if ($pw -ge $MaxPowerW -and $pw -lt 300) { $problems += "GPU power $pw W" }
    }
    $ram = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
    if ($ram -lt $MinFreeRamMiB) { $problems += "free RAM $ram MiB" }
    $disk = [int]((Get-PSDrive C).Free / 1GB)
    if ($disk -lt $MinFreeDiskGB) { $problems += "free disk $disk GB" }

    $jobs = @(Jobs)
    $busy = ($jobs.Count + @(Downloads).Count) -gt 0
    if ($busy -ne $awake) {
        $flags = if ($busy) { $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED } else { $ES_CONTINUOUS }
        [void][Win32.Power]::SetThreadExecutionState($flags)
        $awake = $busy
        Log ("keep-awake " + $(if ($busy) { "ON (job or download running)" } else { "OFF (idle)" }))
    }
    # A limit must be crossed on 2 consecutive checks: nvidia-smi on this laptop has
    # returned a one-off 590 W while idle at 1.5 W (09-27). Real heat builds over tens of
    # seconds, so one extra 3 s check costs no safety.
    $strikes = if ($problems) { $strikes + 1 } else { 0 }
    if ($strikes -eq 1) {
        Log "suspect reading, rechecking: $($problems -join '; ')"
    } elseif ($problems) {
        foreach ($j in $jobs) {
            Stop-Process -Id $j.ProcessId -Force -ErrorAction SilentlyContinue
            Log "KILLED pid $($j.ProcessId) [$($problems -join '; ')] cmd: $($j.CommandLine)"
        }
        if (-not $jobs) { Log "WARN no job running but: $($problems -join '; ')" }
    } elseif ($Once) {
        Log "ok: temp $temp C, vram $vram MiB, power $pw W, ram $ram MiB, disk $disk GB, jobs $($jobs.Count), downloads $(@(Downloads).Count)"
    }
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
}
