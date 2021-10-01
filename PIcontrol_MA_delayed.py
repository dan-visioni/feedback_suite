# Explicit feedback for climate modeling
# Copyright (C) 2020  Ben Kravitz
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Sample control parameters file
#
# Written by Ben Kravitz (bkravitz@iu.edu or ben.kravitz.work@gmail.com)
# Last updated 11 July 2019
#
# This script provides information about the feedback algorithm.  All of this
# is user-customizable.  The other parts of the script will give you outvals,
# lats, lons, and times.  The output of this script should be a list called
# nlvals, which consists of pairs.  The first item in the pair is the name
# of the namelist value.  The second item is the value itself as a string.
#
# This script is written in native python format.  Be careful with brackets [],
# white space, and making sure everything that needs to be a string is actually
# a string by putting it in quotes ''.  All lines beginning with # are comments.

#### USER-SPECIFIED CONTROL PARAMETERS ####

# Target temperature values
start_refvals=[288.81,0.928,-5.85] # initial target values (averages during the years 2030-2049, CESM2-WACCM-MA, SSP2-45)
final_refvals=[288.51,0.880,-5.87] # final target values (averages during the years 2020-2039, same background)

# feedback control gains
kivals=[0.0183,0.0753,0.3120] # taken from Walker's TSMLT controller gain estimates
kpvals=[0.0183,0.0753,0.3120]

# timeline
firstyear=2045       # starting year of simulation
baseyear_start=2040  # "reference year" from which initial targets (start_refvals) are derived
baseyear_final=2030  # "reference year" from which final targets (final_refvals) are derived
x_ramp = 5.0         # defines a range of years over which the feedback is ramped up
transition = 5.0     # defines the length (yrs) of the transition between the initial and final targets

#### USER SPECIFIED CALCULATIONS ####
logfilename='ControlLog_'+runname+'.txt'

logheader=['Timestamp','dT0','sum(dT0)','dT1','sum(dT1)','dT2','sum(dT2)','L0','L1N','L1S','L2','30S(Tg)','15S(Tg)','15N(Tg)','30N(Tg)']

firsttime=0
if os.path.exists(maindir+'/'+logfilename)==False:
    firsttime=1
else:
    loglines=readlog(maindir+'/'+logfilename)

w=makeweights(lats,lons)
T0=numpy.mean(gmean(outvals[0],w))
T1=numpy.mean(l1mean(outvals[0],w,lats))
T2=numpy.mean(l2mean(outvals[0],w,lats))

if firsttime==1:
    timestamp=firstyear
else:
    timestamp=int(loglines[-1][0])+1

dt=timestamp-firstyear     # years of simulation completed

start_refvals = numpy.array(start_refvals)
final_refvals = numpy.array(final_refvals)

if dt<transition:
    trans_fact=dt/transition
    refvals=(1.0-trans_fact)*start_refvals+trans_fact*final_refvals # if we're in the transition period, compute targets for this year
else:
    refvals=final_refvals                                           # if we're past the transition period, targets = final targets

de=numpy.array([T0-refvals[0],T1-refvals[1],T2-refvals[2]]) # error terms

if firsttime==1:
    sumde=de
    sumdt2=de[2]
else:
    sumdt0=float(loglines[-1][2])+(T0-refvals[0])
    sumdt1=float(loglines[-1][4])+(T1-refvals[1])
    sumdt2=float(loglines[-1][6])+(T2-refvals[2])
    sumde=numpy.array([sumdt0,sumdt1,sumdt2])

# feedforward calculations: account for how much T0 has changed since the reference period
dt_old=timestamp-baseyear_start # years passed since the reference period, if we're using the "initial" targets
dt_new=timestamp-baseyear_final # years passed since the reference period, if we're using the "final" targets

if dt<transition:
    trans_fact=dt/transition
    dt_trans=(1.0-trans_fact)*dt_old+trans_fact*dt_new # years passed since the "effective" reference period, accounting for transition period
else:
    dt_trans=dt_new # if we're past the transition period, we only care about years passed since the "final" reference period

change=0.0273*dt_trans # this is approximately how much T0 changes per year, and therefore how much we want to offset w/ fdfwd
sens=4.1 # sensitivity; change to T0 per unit l0 (this may be small because it came from 10-year runs, but we're dividing by 1.4 later to compensate)

l0hat=change/sens/1.40 # dT0 desired over dT0/dl0, and scaled by 1/1.4
l1hat=0.000
l2hat=0.00
ramp_up = 1.0
if (dt2<x_ramp):
    ramp_up = dt2 / x_ramp

# feedback
l0kp1=(kpvals[0]*de[0]+kivals[0]*sumde[0])*ramp_up
l1kp1=(kpvals[1]*de[1]+kivals[1]*sumde[1]-0.5*l0kp1)*ramp_up
l2kp1=(kpvals[2]*de[2]+kivals[2]*sumde[2]-l0kp1)*ramp_up

# all of the feeds
l0step4=l0kp1+l0hat
l1step4=l1kp1+l1hat
l2step4=l2kp1+l2hat
l0=max(l0step4,0)
l1n=min(max(l1step4,0),l0)
l1s=min(max(-l1step4,0),l0)
l2=min(max(l2step4,0),l0-l1s-l1n)
ell=numpy.array([[l0],[l1n],[l1s],[l2]])
# preventing integrator wind-up
if (l2==(l0-l1s-l1n)):
    sumdt2=sumdt2-(T2-refvals[2])
    sumde[2]=sumdt2

M=numpy.array([[0,30,30,0],[0,0,45,20],[20,45,0,0],[40,0,0,40]])
F=numpy.array([[1,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

q=numpy.dot(numpy.dot(numpy.transpose(M),numpy.linalg.inv(F)),ell)

for k in range(len(q)):
    q[k]=max(q[k],0)

newline=[str(timestamp),str(de[0]),str(sumde[0]),str(de[1]),str(sumde[1]),str(de[2]),str(sumde[2]),str(l0),str(l1n),str(l1s),str(l2),str(q[0])[1:-1],str(q[1])[1:-1],str(q[2])[1:-1],str(q[3])[1:-1]]
if firsttime==1:
    linestowrite=[logheader,newline]
else:
    linestowrite=[]
    for k in range(len(loglines)):
        linestowrite.append(loglines[k])
    linestowrite.append(newline)

writelog(maindir+'/'+logfilename,linestowrite)


#### USER SPECIFIED OUTPUT ####
nlname1="/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_30.6S_180E_0.95x1.25_cANN210319.nc"
nlname2="/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_15.6S_180E_0.95x1.25_cANN210319.nc"
nlname3="/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_15.6N_180E_0.95x1.25_cANN210319.nc"
nlname4="/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_30.6N_180E_0.95x1.25_cANN210319.nc"

nlval1="         'SO2    -> "+str(q[0])[1:-1]+"0*/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_30.6S_180E_0.95x1.25_cANN210319.nc'"+"\n"
nlval2="         'SO2    -> "+str(q[1])[1:-1]+"0*/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_15.6S_180E_0.95x1.25_cANN210319.nc',"+"\n"
nlval3="         'SO2    -> "+str(q[2])[1:-1]+"0*/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_15.6N_180E_0.95x1.25_cANN210319.nc',"+"\n"
nlval4="         'SO2    -> "+str(q[3])[1:-1]+"0*/glade/work/geostrat/injection_files/SO2_geoeng_2020-2100_serial_1Tg_21.9-22.1km_30.6N_180E_0.95x1.25_cANN210319.nc',"+"\n"

nlvals=[nlname1,nlval1,nlname2,nlval2,nlname3,nlval3,nlname4,nlval4]

