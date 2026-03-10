$env:PATH = 'C:\Users\eholmes\.local\bin;C:\Users\eholmes\AppData\Local\pnpm;' + $env:PATH
& "$PSScriptRoot\local-build.ps1" -Unpack -SkipSigning
