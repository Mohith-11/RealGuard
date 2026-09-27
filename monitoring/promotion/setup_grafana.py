import urllib.request, urllib.error, json, base64, os
BASE  = "http://localhost:3000"
creds = base64.b64encode(b"admin:admin123").decode()
HDR   = {"Authorization": "Basic "+creds, "Content-Type": "application/json"}

def api(method, path, body=None):
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(BASE+path, data=data, headers=HDR, method=method)
    try:    return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e: return json.loads(e.read())

prom_uid = next((ds["uid"] for ds in (api("GET","/api/datasources") or []) if ds.get("type")=="prometheus"), "prometheus")
def ds(): return {"type":"prometheus","uid":prom_uid}

def stat(pid,title,expr,x,y,w=6,h=4,unit="short",color="blue",noval="N/A"):
    return {"id":pid,"type":"stat","title":title,"gridPos":{"x":x,"y":y,"w":w,"h":h},
            "datasource":ds(),"options":{"reduceOptions":{"calcs":["lastNotNull"]},"colorMode":"background","graphMode":"none"},
            "fieldConfig":{"defaults":{"unit":unit,"noValue":noval,"color":{"mode":"fixed","fixedColor":color},
                "thresholds":{"mode":"absolute","steps":[{"color":color,"value":None}]},"mappings":[]},"overrides":[]},
            "targets":[{"expr":expr,"refId":"A","datasource":ds()}]}

def ts_panel(pid,title,targets,x,y,w=24,h=7):
    return {"id":pid,"type":"timeseries","title":title,"gridPos":{"x":x,"y":y,"w":w,"h":h},
            "datasource":ds(),"options":{"legend":{"displayMode":"list","placement":"bottom"}},
            "fieldConfig":{"defaults":{"unit":"short"},"overrides":[]},
            "targets":[{"expr":t[0],"legendFormat":t[1],"refId":chr(65+i),"datasource":ds()} for i,t in enumerate(targets)]}

def text(pid,title,content,x,y,w=24,h=3):
    return {"id":pid,"type":"text","title":title,"gridPos":{"x":x,"y":y,"w":w,"h":h},"options":{"mode":"markdown","content":content}}

panels=[
    stat(1,"Predictions Total","prediction_requests_total",0,0,w=4,color="blue"),
    stat(2,"Fraud Detected","fraud_predictions_total",4,0,w=4,color="red"),
    stat(3,"Legitimate","legitimate_predictions_total",8,0,w=4,color="green"),
    stat(4,"Labeled Samples","model_labeled_samples",12,0,w=4,color="teal"),
    stat(5,"Metrics Reliable","model_metrics_reliable",16,0,w=4,noval="Unknown",color="orange"),
    stat(6,"Evals Run","model_performance_reports_total",20,0,w=4,color="purple"),
    ts_panel(10,"Requests & Fraud Rate",[("rate(prediction_requests_total[1m])*60","Requests/min"),("rate(fraud_predictions_total[1m])*60","Fraud/min")],0,4),
    ts_panel(11,"Model Metrics",[("model_precision","Precision"),("model_recall","Recall"),("model_f1_score","F1"),("model_pr_auc","PR-AUC"),("model_accuracy","Accuracy")],0,11),
    stat(20,"Accuracy","model_accuracy",0,18,w=6,unit="percentunit",color="purple"),
    stat(21,"Precision","model_precision",6,18,w=6,unit="percentunit",color="blue"),
    stat(22,"Recall","model_recall",12,18,w=6,unit="percentunit",color="orange"),
    stat(23,"F1","model_f1_score",18,18,w=6,unit="percentunit",color="green"),
    text(30,"Deployment","**To promote a model:** `python -m monitoring.promotion.promoter --candidate <version> --approve`  \n**To rollback:** Set alias back then `docker compose restart realguard-api`",0,22),
]
db={"dashboard":{"id":None,"uid":"rg-overview","title":"RealGuard -- System Overview","tags":["realguard","mlops"],"timezone":"browser","refresh":"10s","schemaVersion":38,"version":0,"panels":panels},"folderUid":"","message":"Phase 16","overwrite":True}
r=api("POST","/api/dashboards/db",db)
print("Dashboard:", r.get("status"), r.get("url",""))
