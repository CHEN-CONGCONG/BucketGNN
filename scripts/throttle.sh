#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

## replace DEV=lo with your card (e.g., eth0)
DEV=lo 
if [ "$1" == "del" ]
then
  tc qdisc del dev $DEV root
fi

# LAN and WAN corresponded to 10Gbps/100Mbps bandwidth and 0.3ms/40ms round trip latency, respectively.
if [ "$1" == "lan" ]
then
 tc qdisc del dev $DEV root
## about 10Gbps
 tc qdisc add dev $DEV root handle 1: tbf rate 10000mbit burst 2048k limit 2048k
## about 0.3ms ping latency(实际测试)
 tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.15msec
fi
if [ "$1" == "wan" ]
then
 tc qdisc del dev $DEV root
## about 100Mbps
 tc qdisc add dev $DEV root handle 1: tbf rate 100mbit burst 1024k limit 1024k
## about 40ms ping latency
 tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 20msec
fi