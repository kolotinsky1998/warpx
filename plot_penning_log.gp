set datafile commentschars "#"
set terminal qt size 1400,900
set grid
set key outside right top
set xlabel "Step"
set ylabel "Count"
set title "Penning discharge diagnostics"
set logscale y
set format y "10^{%L}"

logfile = "log.txt"

plot logfile using 1:2  with lines lw 3 lt 1 title "Ne", \
     logfile using 1:3  with lines lw 3 lt 2 title "Ni", \
     logfile using 1:4  with lines lw 2 lt 3 title "Ne_hot", \
     logfile using 1:5  with lines lw 2 lt 4 title "Ni_hot", \
     logfile using 1:6  with lines lw 2 lt 5 title "Ne_abs_EB", \
     logfile using 1:7  with lines lw 2 lt 6 title "Ni_abs_EB", \
     logfile using 1:8  with lines lw 2 lt 8 title "Ne_inj_hot", \
     logfile using 1:9  with lines lw 2 lt 9 title "Ne_inj_cold"

pause -1
