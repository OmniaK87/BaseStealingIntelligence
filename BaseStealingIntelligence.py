import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from os import environ
from sklearn.metrics import r2_score
from flask import Flask, render_template, send_file, redirect, url_for, request
from io import BytesIO

app=Flask(__name__)

wSBmodels = {}
wSBmodelPredictions = {}
wSBmodelR2 = {}
dfMaster = pd.DataFrame.empty

runSB = 0.2
runCS = -0.423
degrees = [1,2,3,4,5,6]
defaultDegree = 5
defaultPlayer = "Billy Hamilton"
#cyan is reserved for the active player
#red and green for positive and negative BSIR when ploting lines
colors = ['b', 'm', 'k', 'b', 'm', 'y', 'k']


#Web App functions
@app.route('/')
@app.route('/index')
def index():
    selectedDeg = request.args.get('selectedDeg', default=defaultDegree)
    return render_template(
        'index.html', 
        degrees=degrees,
        selectedDeg=selectedDeg)

@app.route('/aboutBSIR')
def aboutBSIR():
    selectedDeg = request.args.get('selectedDeg', default=defaultDegree)
    return render_template(
        'aboutBSIR.html', 
        degrees=degrees,
        selectedDeg=selectedDeg)

@app.route('/BSIRByPlayer')
def BSIRByPlayer():
    selectedDeg = request.args.get('selectedDeg', default=defaultDegree)
    selectedPlayer = request.args.get('selectedPlayer', default=defaultPlayer)
    return render_template(
        'BSIRByPlayer.html', 
        degrees=degrees,
        playerList=dfMaster['Full Name'],
        selectedDeg=selectedDeg,
        selectedPlayer=selectedPlayer)

@app.route('/data', methods=['GET'])
def data():
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    ascendingOrder = request.args.get('ascendingOrder', default=1, type=int)
    sortBy = request.args.get('sortBy', default='Full Name')
    columns = request.args.getlist('columns')
    
    dfpage = dfMaster.sort_values(sortBy, ascending=ascendingOrder)
    dfpage.reset_index(drop=True, inplace=True)
    if "all" in columns or columns == []:
        pass
    else:
        dfpage = dfpage[columns]

    datahtml = (
        dfpage.style
        .set_properties(**{'font-size': '9pt', 'font-family': 'Calibri'})
        .render()
    )

    return render_template('data.html', datahtml=datahtml,
    degrees=degrees,
    playerList=dfMaster['Full Name'],
    columnList=dfMaster.columns,
    sortBy=sortBy)

@app.route('/wSBPlot', methods=['GET', 'POST'])
def wSBPlot():
    masterFig, _ = wSB_master_plot(
            degToPlot=request.args.get('deg', default=-1, type=int), fullName=request.args.get('fullName', default=""))
    img = BytesIO()
    masterFig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/BSIRLinePlot', methods=['GET', 'POST'])
def BSIRLinePlot():
    lineFig, _ = BSIR_line_plot(
            degToPlot=request.args.get('deg', default=5, type=int), fullName=request.args.get('fullName', default="", type=str))
    img = BytesIO()
    lineFig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/BSIRPlot', methods=['GET', 'POST'])
def BSIRPlot():
    masterFig, _ = BSIR_plot(
            degToPlot=request.args.get('deg', default=5, type=int), fullName=request.args.get('fullName', default="", type=str),zeroLine=request.args.get('zeroLine', default=True, type=bool))
    img = BytesIO()
    masterFig.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')








#Testing Flask features
@app.route('/test_request_args', methods=['POST', 'GET'])
def test_request_args():
    name = request.args.get('name', 'bob')
    return name


@app.route('/test_redirect', methods=['POST'])
def test_redirect():
    return redirect(url_for('hello'), code=302)


@app.route('/hello')
def hello():
    return render_template('hello.html')






#Setup Functions
def build_dataframe():
    dfSpeed = pd.read_csv("data/MLBSprintSpeeds.csv")
    dfSpeedNames = pd.DataFrame(dfSpeed[["Player","Speed"]])
    dfSpeedNames['nameLast'], dfSpeedNames['nameFirst'] = dfSpeed['Player'].str.split(', ',1).str
    dfSpeedNames['nameLast'] = dfSpeedNames['nameLast'].str.strip()



    dfbatting = pd.read_csv("data/baseballReference2017Batting.csv")
    dfbatting = dfbatting[dfbatting['yearID'] == 2017]
    lgwSB = (np.sum(dfbatting['SB']) * runSB + np.sum(dfbatting['CS'])*runCS) / (np.sum(dfbatting['H'])+np.sum(dfbatting['BB'])+np.sum(dfbatting['HBP'])-np.sum(dfbatting['IBB']))
    dfbatting = dfbatting[['playerID', 'yearID', 'SB', 'CS', 'H', 'BB', 'HBP', 'IBB', 'PA']]

    dfNames = pd.read_csv("data/LahmanMaster.csv")
    dfNames = dfNames[['playerID', 'nameFirst', 'nameLast']]

    dfMaster = pd.merge(dfNames, dfbatting, on='playerID')
    dfMaster['Speed'] = -1

    for _, speedRow in dfSpeedNames.iterrows():
        match = dfMaster.loc[(dfMaster['nameFirst'] == speedRow.nameFirst)]
        if not match.empty:
            for j, row2 in match.iterrows():
                if row2.nameLast == speedRow.nameLast:
                    dfMaster.loc[j,'Speed'] = speedRow['Speed']

    dfMaster = dfMaster.drop(dfMaster[dfMaster['Speed'] < 0].index)
    dfMaster['wSB'] = dfMaster['SB'] * runSB + dfMaster['CS']*runCS - ((dfMaster['H']+dfMaster['BB']+dfMaster['HBP']-dfMaster['IBB'])*lgwSB)

    dfMaster.insert(0, 'Full Name', dfMaster['nameFirst']+' '+dfMaster['nameLast'])
    
    dfMaster = dfMaster.sort_values('Speed')
    dfMaster.reset_index(drop=True, inplace=True)
    return dfMaster

def build_models():
    if dfMaster.empty:  build_dataframe()
    for deg in degrees:
        wSBmodels[deg] = np.polyfit(dfMaster['Speed'], dfMaster['wSB'], deg)

def build_BSIRs():
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    
    for deg in degrees:
        dfMaster['BSIR_{0}'.format(deg)] = dfMaster['wSB']-dfMaster['Speed'].apply(lambda x: predict(x, wSBmodels[deg]))

def get_predictions():
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    for deg in degrees:
        predictions = []    
        for speed in dfMaster['Speed']:
            predictions.append(predict(speed, wSBmodels[deg]))
        wSBmodelPredictions[deg] = predictions

def get_R2s():
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    if wSBmodelPredictions == {}: get_predictions()
    for deg in degrees:
        wSBmodelR2[deg] = r2_score(dfMaster['wSB'], wSBmodelPredictions[deg])







#Plotting Functions
def wSB_master_plot(degToPlot=-1, toShow=False, zeroLine=False, fullName=""):
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    if wSBmodelPredictions == {}: get_predictions()
    if wSBmodelR2 == {}: get_R2s()
    if degToPlot == -1: degToPlot = degrees
    elif type(degToPlot) == int: degToPlot = [degToPlot]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    if zeroLine: ax.plot(dfMaster['Speed'], np.zeros(len(dfMaster['Speed'])), color='k')
    ax.scatter(dfMaster['Speed'], dfMaster['wSB'], label="Players", c='y', s=(plt.rcParams['lines.markersize'] ** 2)/(len(degToPlot)), alpha = 3/4)
    for deg in degToPlot:
        ax.plot(dfMaster['Speed'], wSBmodelPredictions[deg], label="Degree: {0} R2: {1:0.4f}".format(deg, wSBmodelR2[deg]), color=colors[deg-1], alpha=1/2)

    if fullName != "":
        player = find_player(fullName)
        if not player.empty: 
            for _, row in player.iterrows():
                ax.scatter(row['Speed'], row['wSB'], label="{0} BSIR".format(fullName), color='C', s=(plt.rcParams['lines.markersize'] ** 2)*2)
                ax.text(23, 2, "{0}\nwSB: {1:.4f}".format(fullName, row['wSB']), fontweight='bold', fontsize=20)
                ax.arrow(23, 2, row['Speed']-23, row['wSB']-2)
        else:
            print("player not found")

    ax.set_xlabel("Speed", fontsize=16)
    ax.set_ylabel("wSB", fontsize=16)
    ax.set_title("wSB for all Players by Speed", fontsize=20); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    if toShow: plt.show()
    return fig, ax



def BSIR_plot(degToPlot=defaultDegree, movingAverage=1, toShow=False, zeroLine=False, fullName=""):
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    if 'BSIR_{0}'.format(degToPlot) not in dfMaster.columns: build_BSIRs()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    if zeroLine: ax.plot(dfMaster['Speed'], np.zeros(len(dfMaster['Speed'])), color='k')

    ax.scatter(dfMaster['Speed'], dfMaster['BSIR_{0}'.format(degToPlot)], label="Players BSIR_{0}".format(degToPlot), color='b', s=(plt.rcParams['lines.markersize'] ** 2)/2, alpha=3/4)
    if movingAverage > 1:
        BSIR_MA = movingaverage(dfMaster['BSIR_{0}'.format(degToPlot)],movingAverage)
        ax.plot(dfMaster['Speed'], BSIR_MA, label="BSIR_{0} Trend".format(degToPlot), color=colors[degToPlot-1], lw=(plt.rcParams['lines.markersize'] ** 2)/8)
    
    if fullName:
        player = find_player(fullName)
        if not player.empty: 
            for _, row in player.iterrows():
                ax.scatter(row['Speed'], row['BSIR_{0}'.format(degToPlot)], label="{0} BSIR".format(fullName), color='c', s=(plt.rcParams['lines.markersize'] ** 2)*2)
                ax.text(23, 2, "{0}\nBSIR_{2}: {1:.3f}\nwSB: {3:.3f}".format(fullName, row['BSIR_{0}'.format(degToPlot)], degToPlot, row['wSB']), fontweight='bold', fontsize=20)
                ax.arrow(23, 2, row['Speed']-23, row['BSIR_{0}'.format(degToPlot)]-2)
        else:
            print("player not found")

    ax.set_xlabel("Speed", fontsize=16)
    ax.set_ylabel("BSIR (Base Running Intelligence Runs)", fontsize=16)
    ax.set_title("BSIR from {0}th degree averages".format(degToPlot), fontsize=20); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    if toShow: plt.show()
    return fig, ax



def BSIR_line_plot(degToPlot=defaultDegree, toShow=False, zeroLine=False, fullName=""):
    if dfMaster.empty: build_dataframe()
    if wSBmodels == {}: build_models()
    if wSBmodelPredictions == {}: get_predictions()
    if wSBmodelR2 == {}: get_R2s()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    if zeroLine: ax.plot(dfMaster['Speed'], np.zeros(len(dfMaster['Speed'])), color='k', alpha=1/2)
    ax.scatter(dfMaster['Speed'], dfMaster['wSB'], label="Players", c='k', s=(plt.rcParams['lines.markersize'] ** 2)/2, alpha=1/4)

    ax.plot(dfMaster['Speed'], wSBmodelPredictions[degToPlot], label="Degree: {0} R2: {1:0.4f}".format(degToPlot, wSBmodelR2[degToPlot]))
    for _,row in dfMaster.iterrows():
        prediction = predict(row['Speed'], wSBmodels[degToPlot])
        lineColor='r'
        if row['wSB'] - prediction > 0: lineColor='g'
        ax.plot((row['Speed'],row['Speed']), (row['wSB'], prediction), color=lineColor)
    
    if fullName:
        player = find_player(fullName)
        if not player.empty: 
            for _, row in player.iterrows():
                prediction = predict(row['Speed'], wSBmodels[degToPlot])
                ax.plot((row['Speed'],row['Speed']), (row['wSB'], prediction), label="{0} BSIR".format(fullName), color='c', lw=4)
                ax.scatter(row['Speed'], row['wSB'], label="{0} BSIR".format(fullName), color='C', s=(plt.rcParams['lines.markersize'] ** 2)*2)
                ax.text(24, 3, "{0}\nBSIR_{2}: {1:.3f}\nwSB:{3:.3f}".format(fullName, row['BSIR_{0}'.format(degToPlot)], degToPlot, row['wSB']), fontweight='bold', fontsize=20)
                ax.arrow(24, 3, row['Speed']-24, row['wSB']-3)
        else:
            print("player not found")
    
    ax.set_xlabel("Speed", fontsize=16)
    ax.set_ylabel("wSB", fontsize=16)
    ax.set_title("BSIR_{0} Visualized".format(degToPlot), fontsize=20); ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    if toShow: plt.show()
    return fig, ax



#Helper Functions
def predict(speed, params):
    wSB = 0
    indexes = list(range(len(params)-1,-1,-1))
    for power, index in zip(indexes, reversed(indexes)):
        wSB += (speed**power)*params[index]
    return wSB

def find_player(fullName):
    if dfMaster.empty: build_dataframe()
    fullMatch = dfMaster.loc[dfMaster['Full Name'] == fullName]
    return pd.DataFrame(fullMatch)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    forwardsma = np.convolve(values, weights, 'valid')
    #perform a moving average backwards, then reverse it back
    backwardsma = np.convolve(values[::-1], weights, 'valid')[::-1]
    

    dualDirectionMA = np.zeros(len(values))
    dualDirectionMA[len(dualDirectionMA)-window+1:] = forwardsma[len(forwardsma)-window+1:]
    dualDirectionMA[:window-1] = backwardsma[:window-1]
    dualDirectionMA[window-1:len(dualDirectionMA)-window+1] = [(a+b)/2 for a,b in zip(forwardsma[:len(forwardsma)-window+1], backwardsma[window-1:])]
    return dualDirectionMA

 
if __name__ == "__main__":
    dfMaster = build_dataframe()
    build_models()
    build_BSIRs()
    port = int(environ.get("PORT", 33507))
    app.run(host='0.0.0.0', port=port)

