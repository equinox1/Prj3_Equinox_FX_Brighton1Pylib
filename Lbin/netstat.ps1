$range = 8000..9000
$netstat = netstat -ano | Select-String "TCP|UDP"

foreach ($line in $netstat) {
    foreach ($port in $range) {
        if ($line -match "[:\.]$port\s") {
            $line
            break
        }
    }
}

