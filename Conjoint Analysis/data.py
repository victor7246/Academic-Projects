import pandas
import numpy
data = pandas.read_csv("data/data.csv")
l = []
for i in range(len(data)):
	if data.iloc[i][0] is not numpy.nan:
		''' l.append((i+1,data.iloc[i][0],data.iloc[i][1],apple,samsung,sony,cl,cm,ch,pl,pm,ph,sl,sm,sh,ml,mm,mh,pl,pm,ph,pvh,data.iloc[i][2])) '''
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,1,0,data.iloc[i][2]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,data.iloc[i][3]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,1,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,0,0,data.iloc[i][4]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,data.iloc[i][5]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,data.iloc[i][6]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,0,1,data.iloc[i][7]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,0,1,0,0,data.iloc[i][8]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,data.iloc[i][9]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,data.iloc[i][10]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],1,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,data.iloc[i][11]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,data.iloc[i][12]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,data.iloc[i][13]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,1,data.iloc[i][14]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0,0,0,data.iloc[i][15]))
		l.append((i+1,data.iloc[i][0],data.iloc[i][1],1,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,0,data.iloc[i][16]))

pandas.DataFrame(l).to_csv("data/main.csv")
