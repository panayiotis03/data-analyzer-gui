import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import tempfile

from datetime import datetime,date


#  CONFIGURATION

ROOM_STRUCTURE = {
    "TOFIS":  ["standard"],
    "DRAKOS": ["Front", "Back"],
    "TASOS":  ["Front_Left", "Front_Right", "Back_Left", "Back_Right"],
}

TASOS_SENSOR_FOLDER = {
    "Front_Left":  "TASOS_front_left",
    "Front_Right": "TASOS_front_left",
    "Back_Left":   "TASOS_back_right",
    "Back_Right":  "TASOS_back_right",
}

MONTH_FILE_ALIASES = {
    '09': ['sep', 'sept', 'september'],
    '10': ['oct', 'october'],
    '11': ['nov', 'november'],
    '12': ['dec', 'december'],
    '01': ['jan', 'january', 'junu', 'june', 'juan'],
}

MONTH_FEEDBACK_NAME = {
    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec', '01': 'Jan',
}

FEEDBACK_WEIGHTS = {
    'comfortable': 1.0, 'pleasant':   1.0,
    'neutral':     0.5, 'noticeable': 0.5,
    'irritating':  0.0, 'too hot':    0.0,
    'too cold':    0.0, 'too dry':    0.0,
    'too humid':   0.0, 'unpleasant': 0.0,
}

#  FOLDER SCAN
def scan_folder_inventory(base_path):
    sensors = []; feedbacks = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            if fname.lower().endswith('.csv'):
                p = os.path.join(root, fname)
                if 'feedback' in fname.lower():
                    feedbacks.append(p)
                else:
                    sensors.append(p)
    return sorted(sensors), sorted(feedbacks)

#  FILE FINDERS  
def find_sensor_path(base_path, room, part, month_num):
    aliases = MONTH_FILE_ALIASES.get(month_num, [])

    if room == "TOFIS":
        search_dirs = [
            os.path.join(base_path, "Tofis"),
            os.path.join(base_path, "tofis"),
            base_path,
        ]
    elif room == "DRAKOS":
        pl = part.lower()
        search_dirs = [
            os.path.join(base_path, "drakos", f"drakos {pl}"),
            os.path.join(base_path, "drakos", f"drakos_{pl}"),
            os.path.join(base_path, "drakos"),
            base_path,
        ]
    elif room == "TASOS":
        sf = TASOS_SENSOR_FOLDER.get(part, f"TASOS_{part.lower()}")
        search_dirs = [
            os.path.join(base_path, "tasos", sf),
            os.path.join(base_path, "tasos", sf.lower()),
            os.path.join(base_path, "tasos"),
            base_path,
        ]
    else:
        search_dirs = [base_path]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.lower().endswith('.csv'):
                continue
            if 'feedback' in fname.lower():
                continue
            name_noext = os.path.splitext(fname)[0].lower()
            if name_noext in aliases:
                return os.path.join(d, fname)
    return None


def find_feedback_path(base_path, room, part, month_num):
    fb_month = MONTH_FEEDBACK_NAME.get(month_num, '')

    if room == "TOFIS":
        candidates = [f"TOFIS_{fb_month}_feedbacks.csv"]
        search_dirs = [
            os.path.join(base_path, "Tofis"),
            os.path.join(base_path, "tofis"),
            base_path,
        ]
    elif room == "DRAKOS":
        candidates = [f"DRAKOS_{part}_{fb_month}_feedbacks.csv"]
        search_dirs = [os.path.join(base_path, "drakos"), base_path]
    elif room == "TASOS":
        candidates = [f"TASOS_{part}_{fb_month}_feedbacks.csv"]
        search_dirs = [os.path.join(base_path, "tasos"), base_path]
    else:
        candidates = []; search_dirs = [base_path]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.lower() in [c.lower() for c in candidates]:
                return os.path.join(d, fname)
    return None


def read_csv_path(path):
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return pd.read_csv(path, sep=None, engine='python', encoding=enc)
        except Exception:
            continue
    return None

#  SEASONAL CONTEXT
def get_seasonal_context(d):
    if date(d.year, 9, 1) <= d <= date(d.year, 11, 20):
        return "ΠΕΡΙΟΔΟΣ 1"
    if d >= date(d.year, 11, 21) or d <= date(d.year, 1, 31):
        return "ΠΕΡΙΟΔΟΣ 2"
    return "ΕΚΤΟΣ ΠΕΡΙΟΔΟΥ"

#  DECISION TREES
def decision_tree_p1(v, comfort_pct, total_f):
    r = []
    T = v.get('T')
    if T in (None, "N/A"):   
        r.append(('info', "Θερμοκρασία", "Δεν υπάρχουν δεδομένα."))
    elif T < 23:             
        r.append(('warn', f"Θερμοκρασία ({T} °C)", "Μείωση ή κλείσιμο κλιματισμού (A/C) και κλείσιμο παραθύρων."))
    elif T <= 27:            
        r.append(('ok', f"Θερμοκρασία ({T} °C)", "Βέλτιστη για Π1 (23–27°C)."))
    else:                    
        r.append(('alert', f"Θερμοκρασία ({T} °C)", "Ενεργοποίηση ή εντατικοποίηση A/C (Ψύξη) και κλείσιμο παραθύρων."))

    H = v.get('H')
    if H in (None, "N/A"):   
        r.append(('info', "Υγρασία", "Δεν υπάρχουν δεδομένα."))
    elif H < 30:             
        r.append(('alert', f"Υγρασία ({H} %)", "Πολύ ξηρός αέρας. Κλείσιμο A/C, ελαφρύ άνοιγμα παραθύρων για είσοδο φυσικής υγρασίας."))
    elif H < 40:             
        r.append(('warn', f"Υγρασία ({H} %)", "Ελαφρά χαμηλή. Μείωση έντασης A/C ή σύντομος φυσικός αερισμός από παράθυρα."))
    elif H <= 60:            
        r.append(('ok', f"Υγρασία ({H} %)", "Βέλτιστη για Π1 (40–60%)."))
    elif H <= 65:            
        r.append(('warn', f"Υγρασία ({H} %)", "Ελαφρώς αυξημένη. Ενεργοποίηση A/C σε λειτουργία Αφύγρανσης (Dry Mode)."))
    else:                    
        r.append(('alert', f"Υγρασία ({H} %)", "Υψηλή υγρασία. Λειτουργία A/C σε Dry Mode. Αποφύγετε το άνοιγμα παραθύρων αν βρέχει έξω."))

    C = v.get('C')
    if C in (None, "N/A"):   
        r.append(('info', "CO2", "Δεν υπάρχουν δεδομένα."))
    elif C < 600:            
        r.append(('ok', f"CO2 ({C} ppm)", "Εξαιρετική ποιότητα αέρα."))
    elif C <= 1000:          
        r.append(('ok', f"CO2 ({C} ppm)", "Καλό επίπεδο για Π1."))
    elif C <= 1500:          
        r.append(('warn', f"CO2 ({C} ppm)", "Αύξηση μηχανικού εξαερισμού ή ελαφρύ άνοιγμα παραθύρων/πόρτας."))
    else:                    
        r.append(('alert', f"CO2 ({C} ppm)", "Άμεσος αερισμός – Μέγιστη σκάλα εξαερισμού και άνοιγμα παραθύρων & πορτών."))

    VOC = v.get('VOC')
    if VOC in (None, "N/A"): 
        r.append(('info', "VOC", "Δεν υπάρχουν δεδομένα."))
    else:
        ug = round(VOC * 3, 0)
        if ug < 300:         
            r.append(('ok', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Αποδεκτή ποιότητα."))
        elif ug <= 500:      
            r.append(('warn', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Αύξηση αερισμού μέσω μηχανικού εξαερισμού ή παραθύρων."))
        elif ug <= 1000:     
            r.append(('warn', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Έντονος αερισμός – Λειτουργία εξαερισμού και άνοιγμα παραθύρων/πορτών."))
        else:                
            r.append(('alert', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Άμεσος αερισμός (Εξαερισμός & Παράθυρα) και απομάκρυνση χημικών πηγών."))
    return r


def decision_tree_p2(v, comfort_pct, total_f):
    r = []
    T = v.get('T')
    if T in (None, "N/A"):   
        r.append(('info', "Θερμοκρασία", "Δεν υπάρχουν δεδομένα."))
    elif T < 20:             
        r.append(('alert', f"Θερμοκρασία ({T} °C)", "Χαμηλή θερμοκρασία. Ενεργοποίηση A/C στη θέρμανση και κλείσιμο παραθύρων."))
    elif T <= 24:            
        r.append(('ok', f"Θερμοκρασία ({T} °C)", "Βέλτιστη για Π2 (20–24°C)."))
    else:                    
        r.append(('warn', f"Θερμοκρασία ({T} °C)", "Αυξημένη θερμοκρασία. Μείωση θέρμανσης ή ενεργοποίηση A/C στην ψύξη."))

    H = v.get('H')
    if H in (None, "N/A"):   
        r.append(('info', "Υγρασία", "Δεν υπάρχουν δεδομένα."))
    elif H < 30:             
        r.append(('alert', f"Υγρασία ({H} %)", "Πολύ χαμηλή υγρασία. Αν βρέχει έξω, ανοίξτε ελαφρώς τα παράθυρα. Κλείστε το A/C αν είναι εφικτό."))
    elif H <= 50:            
        r.append(('ok', f"Υγρασία ({H} %)", "Βέλτιστη για Π2 (30–50%)."))
    elif H <= 60:            
        r.append(('warn', f"Υγρασία ({H} %)", "Αυξημένη υγρασία. Ενεργοποίηση A/C σε λειτουργία Αφύγρανσης (Dry) και χρήση εξαερισμού."))
    else:                    
        r.append(('alert', f"Υγρασία ({H} %)", "Υψηλή υγρασία. Λειτουργία A/C αποκλειστικά σε Dry Mode. Κλείσιμο παραθύρων αν βρέχει έξω."))

    C = v.get('C')
    if C in (None, "N/A"):   
        r.append(('info', "CO2", "Δεν υπάρχουν δεδομένα."))
    elif C < 1000:          
        r.append(('ok', f"CO2 ({C} ppm)", "Καλό επίπεδο για Π2."))
    elif C <= 1200:          
        r.append(('ok', f"CO2 ({C} ppm)", "Αποδεκτό επίπεδο χειμερινών συνθηκών."))
    elif C <= 1500:          
        r.append(('warn', f"CO2 ({C} ppm)", "Ανάγκη ανανέωσης αέρα. Ενεργοποίηση μηχανικού εξαερισμού."))
    else:                    
        r.append(('alert', f"CO2 ({C} ppm)", "Κρίσιμο CO2. Μέγιστη λειτουργία εξαερισμού και άνοιγμα πόρτας/παραθύρων για γρήγορο καθαρισμό."))

    VOC = v.get('VOC')
    if VOC in (None, "N/A"): 
        r.append(('info', "VOC", "Δεν υπάρχουν δεδομένα."))
    else:
        ug = round(VOC * 3, 0)
        if ug < 300:         
            r.append(('ok', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Αποδεκτή ποιότητα."))
        elif ug <= 500:      
            r.append(('warn', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Αύξηση αερισμού μέσω του μηχανικού εξαερισμού."))
        elif ug <= 1000:     
            r.append(('warn', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Ενεργοποίηση εξαερισμού στη μέγιστη σκάλα και άνοιγμα πόρτας."))
        else:                
            r.append(('alert', f"VOC ({VOC} ppb ≈ {ug:.0f} μg/m³)", "Άμεση ενεργοποίηση εξαερισμού, άνοιγμα παραθύρων/πορτών και εντοπισμός εσωτερικών πηγών."))
    return r


def decision_tree_common(v, comfort_pct, total_f, P1):
    r = []
    PM1 = v.get('PM1')
    if PM1 in (None, "N/A"): 
        r.append(('info', "PM1", "Δεν υπάρχουν δεδομένα."))
    elif PM1 < 10:           
        r.append(('ok', f"PM1 ({PM1} μg/m³)", "Καθαρός αέρας."))
    elif PM1 <= 20:          
        r.append(('warn', f"PM1 ({PM1} μg/m³)", "Ελαφρά αυξημένα σωματίδια. Χρήση μηχανικού εξαερισμού (αν έχει φίλτρα) ή κλείσιμο παραθύρων αν η πηγή είναι εξωτερική."))
    else:                    
        r.append(('alert', f"PM1 ({PM1} μg/m³)", "Υψηλά σωματίδια. Κλείσιμο παραθύρων (αποφυγή εξωτερικής ρύπανσης), λειτουργία A/C για ανακυκλοφορία αέρα."))

    PM25 = v.get('PM25')
    if PM25 in (None, "N/A"):
        r.append(('info', "PM2.5", "Δεν υπάρχουν δεδομένα."))
    elif PM25 < 12:          
        r.append(('ok', f"PM2.5 ({PM25} μg/m³)", "Καλή ποιότητα αέρα."))
    elif PM25 <= 25:         
        r.append(('warn', f"PM2.5 ({PM25} μg/m³)", "Μέτρια ποιότητα. Έλεγχος αν η σκόνη έρχεται από έξω (κλείσιμο παραθύρων) και χρήση εξαερισμού."))
    else:                    
        r.append(('alert', f"PM2.5 ({PM25} μg/m³)", "Υψηλή συγκέντρωση. Κλείσιμο παραθύρων, ενεργοποίηση A/C και μηχανικού εξαερισμού για φιλτράρισμα."))

    N = v.get('N')
    if N in (None, "N/A"):   
        r.append(('info', "Θόρυβος", "Δεν υπάρχουν δεδομένα."))
    elif N < 35:             
        r.append(('ok', f"Θόρυβος ({N} dBA)", "Ήσυχο περιβάλλον."))
    elif N <= 50:            
        r.append(('warn', f"Θόρυβος ({N} dBA)", "Κλείσιμο παραθύρων για μείωση εξωτερικού θορύβου και χρήση A/C/εξαερισμού για κλιματισμό."))
    else:                    
        r.append(('alert', f"Θόρυβος ({N} dBA)", "Υψηλός θόρυβος. Κλείσιμο όλων των ανοιγμάτων (παράθυρα/πόρτες) και εντοπισμός εσωτερικών πηγών."))

    Pv = v.get('P')
    if Pv in (None, "N/A"):  
        r.append(('info', "Πίεση", "Δεν υπάρχουν δεδομένα."))
    elif 980 <= Pv <= 1050: 
        r.append(('ok', f"P ({Pv} hPa)", "Κανονική ατμοσφαιρική πίεση."))
    else:                    
        r.append(('alert', f"Πίεση ({Pv} hPa)", "Ατμοσφαιρική πίεση εκτός φυσιολογικών ορίων. Πιθανή δυσλειτουργία αισθητήρα ή ακραίο καιρικό φαινόμενο. Συνιστάται έλεγχος και επαναβαθμονόμηση του σταθμού."))

    if total_f >= 1 and comfort_pct is not None:
        if comfort_pct < 40:   
            r.append(('alert', f"Satisfaction ({comfort_pct:.1f}%)", "Χαμηλή άνεση χρηστών! Ρυθμίστε το A/C και τον αερισμό βάσει των παραπάνω ενδείξεων."))
        elif comfort_pct < 65: 
            r.append(('warn', f"Satisfaction ({comfort_pct:.1f}%)", "Μέτρια άνεση, συνιστάται μικρή μικρορύθμιση θερμοκρασίας/εξαερισμού."))
        else:                  
            r.append(('ok', f"Satisfaction ({comfort_pct:.1f}%)", "Υψηλή θερμική άνεση και ικανοποίηση στην αίθουσα."))
    return r

#  CHARTS
def chart_dual_axis(v, comfort_pct, target_time, season_id):
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    labels = ['Θερμ.\n(°C)','Υγρασία\n(%)','CO2\n(÷10)','PM2.5\n(μg/m³)','Θόρυβος\n(dBA)']
    raw    = [v.get('T'), v.get('H'), v.get('C'), v.get('PM25'), v.get('N')]
    any_data = any(x not in (None,"N/A") for x in raw)
    values = [(x/10 if i==2 else x) if x not in (None,"N/A") else 0 for i,x in enumerate(raw)]
    colors = ['#E74C3C','#3498DB','#27AE60','#9B59B6','#F39C12']

    fig, ax1 = plt.subplots(figsize=(9,4))
    fig.patch.set_facecolor('#F8F9FA'); ax1.set_facecolor('#FFFFFF')

    if any_data:
        bars = ax1.bar(labels, values, color=colors, alpha=0.82, width=0.5,
                       edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, raw):
            if val not in (None,"N/A"):
                ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                         f'{val:.1f}' if isinstance(val,float) else str(val),
                         ha='center', va='bottom', fontsize=7.5, color='#2C3E50')
            else:
                ax1.text(bar.get_x()+bar.get_width()/2, 1.0,
                         'N/A', ha='center', va='bottom', fontsize=8,
                         color='#95A5A6', style='italic')
        max_val = max((x for x in values if x), default=10)
        ax1.set_ylim(0, max(max_val * 1.45, 20))
    else:
        # Δεν υπάρχουν sensor δεδομένα,τοτε N/A
        ax1.bar(labels, [5]*5, color=[c+'55' for c in colors],
                width=0.5, edgecolor='white', linewidth=0.8)
        for i, lbl in enumerate(labels):
            ax1.text(i, 6, 'N/A', ha='center', va='bottom', fontsize=9,
                     color='#7F8C8D', fontweight='bold')
        ax1.set_ylim(0, 40)
        ax1.text(0.5, 0.6,
                 '⚠  Δεν βρέθηκαν δεδομένα αισθητήρα\nγια αυτό το χρονικό παράθυρο (±90 λεπτά)',
                 transform=ax1.transAxes, ha='center', va='center',
                 fontsize=10, color='#E74C3C',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#FDEDEC',
                           edgecolor='#E74C3C', alpha=0.9))

    ax1.set_ylabel('Τιμή αισθητήρα', fontsize=9, color='#2C3E50')
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Satisfaction line 
    ax2 = ax1.twinx()
    sat = comfort_pct if comfort_pct is not None else 0
    cs  = '#27AE60' if sat>=65 else ('#F39C12' if sat>=40 else '#E74C3C')
    sat_label = f'Satisfaction: {sat:.1f}%' if comfort_pct is not None else 'Satisfaction: N/A'
    ax2.axhline(y=sat, color=cs, linestyle='--', linewidth=2.5, label=sat_label)
    ax2.fill_between(range(len(labels)), sat, alpha=0.10, color=cs)
    ax2.set_ylabel('Satisfaction (%)', fontsize=9, color=cs)
    ax2.set_ylim(0, 115)
    ax2.tick_params(axis='y', labelcolor=cs, labelsize=8)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.spines['top'].set_visible(False)

    ps = 'Π1' if 'ΠΕΡΙΟΔΟΣ 1' in season_id else 'Π2'
    ax1.set_title(f"Sensor Data & Satisfaction  |  {target_time.strftime('%d/%m/%Y %H:%M')}  |  {ps}",
                  fontsize=10, pad=10, color='#2C3E50', fontweight='bold')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


def chart_heatmap(v, category_counts, total_f):
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    NEGATIVE={'too hot','too cold','too dry','too humid','irritating','unpleasant'}
    NEUTRAL={'neutral','noticeable'}; POSITIVE={'comfortable','pleasant'}
    tot=max(total_f,1)
    pos=sum(c for l,c in category_counts.items() if l.lower() in POSITIVE)/tot
    neu=sum(c for l,c in category_counts.items() if l.lower() in NEUTRAL)/tot
    neg=sum(c for l,c in category_counts.items() if l.lower() in NEGATIVE)/tot
    params=['Θερμοκρασία','Υγρασία','CO2','VOC','PM1','PM2.5','Θόρυβος']
    pvals=[v.get('T','N/A'),v.get('H','N/A'),v.get('C','N/A'),
           v.get('VOC','N/A'),v.get('PM1','N/A'),v.get('PM25','N/A'),v.get('N','N/A')]
    fbl=['Θετικό\n(Comf.)','Ουδέτερο\n(Neutral)','Αρνητικό\n(Hot/Cold…)']
    matrix=np.array([[pos,neu,neg]]*len(params))
    fig,ax=plt.subplots(figsize=(8,4.5))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#FFFFFF')
    im=ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    cbar=plt.colorbar(im,ax=ax,shrink=0.85,pad=0.02)
    cbar.set_label('Ποσοστό feedback',fontsize=8); cbar.ax.tick_params(labelsize=7)
    ax.set_xticks(range(len(fbl))); ax.set_xticklabels(fbl,fontsize=8.5)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([f"{p}  [{sv if sv!='N/A' else '—'}]" for p,sv in zip(params,pvals)],fontsize=8.5)
    for i in range(len(params)):
        for j,val in enumerate([pos,neu,neg]):
            tc='white' if val>0.65 or val<0.15 else '#1A1A1A'
            ax.text(j,i,f'{val*100:.0f}%',ha='center',va='center',fontsize=9,color=tc,fontweight='bold')
    ax.set_title('Correlation Heatmap – Sensor Parameters vs Student Feedback',
                 fontsize=10,pad=10,color='#2C3E50',fontweight='bold')
    ax.spines[:].set_visible(False); plt.tight_layout(pad=1.5)
    buf=io.BytesIO(); plt.savefig(buf,dpi=150,bbox_inches='tight',facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


def chart_pie(category_counts, total_f, comfort_pct):
    if not category_counts or total_f==0: return None
    matplotlib.rcParams['font.family']='DejaVu Sans'
    cmap={'comfortable':'#27AE60','pleasant':'#2ECC71','neutral':'#F39C12',
          'noticeable':'#E67E22','too hot':'#E74C3C','too cold':'#3498DB',
          'too dry':'#E67E22','too humid':'#1ABC9C','irritating':'#C0392B','unpleasant':'#8E44AD'}
    labels=list(category_counts.keys()); sizes=list(category_counts.values())
    colors=[cmap.get(l.lower(),'#95A5A6') for l in labels]
    fig,ax=plt.subplots(figsize=(5.5,4)); fig.patch.set_facecolor('#F8F9FA')
    _,_,ats=ax.pie(sizes,labels=labels,colors=colors,explode=[0.04]*len(labels),
                   autopct='%1.1f%%',startangle=140,
                   textprops={'fontsize':8.5},wedgeprops={'edgecolor':'white','linewidth':1.5})
    for at in ats: at.set_fontsize(8); at.set_color('white'); at.set_fontweight('bold')
    ax.set_title(f'Κατανομή Feedback | Satisfaction: {comfort_pct:.1f}% | N={total_f}',
                 fontsize=9.5,pad=10,color='#2C3E50',fontweight='bold')
    plt.tight_layout()
    buf=io.BytesIO(); plt.savefig(buf,dpi=150,bbox_inches='tight',facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf

#  MAIN ANALYSIS
def run_analysis(base_path, room, part, sel_date, hour, minute):
    season_id = get_seasonal_context(sel_date)
    month_num = sel_date.strftime('%m')
    target_time = pd.Timestamp(year=sel_date.year, month=sel_date.month, day=sel_date.day,
                               hour=hour, minute=minute)
    s_start = target_time - pd.Timedelta(minutes=90)
    s_end   = target_time + pd.Timedelta(minutes=90)
    f_start = target_time - pd.Timedelta(minutes=90)
    f_end   = target_time + pd.Timedelta(minutes=90)

    s_path = find_sensor_path(base_path, room, part, month_num)
    f_path = find_feedback_path(base_path, room, part, month_num)

    res = {
        'season_id':season_id, 'target_time':target_time,
        's_path':s_path, 'f_path':f_path, 'error':None,
        'v':{}, 'comfort_pct':None, 'category_counts':{}, 'total_f':0,
        'decisions_main':[], 'decisions_common':[],
        'P1':'ΠΕΡΙΟΔΟΣ 1' in season_id,
        's_start':s_start,'s_end':s_end,'f_start':f_start,'f_end':f_end,
        'fb_col':None,'room':room,'part':part,
    }

    if not s_path or not f_path:
        res['error']='missing_files'; return res

    df_s = read_csv_path(s_path)
    df_f = read_csv_path(f_path)
    if df_s is None or df_f is None:
        res['error']='read_error'; return res

    df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
    df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')

    snap     = df_s[(df_s['Timestamp']>=s_start)&(df_s['Timestamp']<=s_end)]
    window_f = df_f[(df_f['Timestamp']>=f_start)&(df_f['Timestamp']<=f_end)].copy()

    if snap.empty:
        res['error']='no_sensor_data'; return res

    m_col = next((c for c in snap.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                 next((c for c in snap.columns if 'meas' in c.lower()),None))
    v_col = next((c for c in snap.columns if c.lower()=='value'),
                 next((c for c in snap.columns if 'valu' in c.lower()),None))

    if not m_col or not v_col:
        res['error']=f'no_columns: {list(snap.columns)}'; return res

    def get_val(names):
        mask=snap[m_col].astype(str).str.strip().str.lower().isin([n.lower() for n in names])
        vals=pd.to_numeric(snap.loc[mask,v_col],errors='coerce').dropna()
        return round(vals.mean(),1) if not vals.empty else "N/A"

    v = {
        'T':   get_val(['Temperature']),
        'H':   get_val(['Humidity']),
        'C':   get_val(['Carbon Dioxide','CO2']),
        'VOC': get_val(['VOC','Volatile Organic Compounds']),
        'PM1': get_val(['PM1','PM 1','PM1.0','pm1.0']),
        'PM25':get_val(['PM2.5','PM 2.5','pm2.5']),
        'N':   get_val(['Noise']),
        'P':   get_val(['Pressure']),
    }
    res['v'] = v

    total_f=len(window_f); res['total_f']=total_f
    comfort_pct=None; fb_col=None; category_counts={}

    if total_f>0:
        fb_col='Temperature_Feedback' if 'Temperature_Feedback' in window_f.columns else None
        if not fb_col:
            fb_col=next((c for c in window_f.columns
                         if any(k in c.lower() for k in ('temp','feed','comfort'))),None)
        if fb_col:
            norm=window_f[fb_col].astype(str).str.strip()
            category_counts=norm.value_counts().to_dict()
            ws=sum(FEEDBACK_WEIGHTS.get(l.lower(),0.0)*cnt for l,cnt in category_counts.items())
            comfort_pct=round((ws/total_f)*100,1)

    res.update({'comfort_pct':comfort_pct,'category_counts':category_counts,'fb_col':fb_col})
    P1=res['P1']
    res['decisions_main']  =(decision_tree_p1 if P1 else decision_tree_p2)(v,comfort_pct,total_f)
    res['decisions_common']= decision_tree_common(v,comfort_pct,total_f,P1)
    return res

#  REPORT TEXT
def build_report_text(r):
    lines=[]; a=lines.append
    a(f"  DOMOGNOSTICS ANALYSIS | {r['target_time'].strftime('%d/%m/%Y %H:%M')}")
    a(f"  CLIMATE CONTEXT: {r['season_id']}")
    a("="*65)
    a(f"  Sensor CSV   : {os.path.basename(r['s_path']) if r['s_path'] else 'ΔΕΝ ΒΡΕΘΗΚΕ'}")
    a(f"  Feedback CSV : {os.path.basename(r['f_path']) if r['f_path'] else 'ΔΕΝ ΒΡΕΘΗΚΕ'}")
    a("="*65)
    v=r['v']
    a(f"\n  1. SENSOR DATA  (±90 λεπτά | {r['s_start'].strftime('%H:%M')} – {r['s_end'].strftime('%H:%M')})")
    a("-"*65)
    for lbl,key,unit in [("Θερμοκρασία","T","°C"),("Υγρασία","H","%"),
                          ("CO2","C","ppm"),("VOC","VOC","ppb"),
                          ("PM1","PM1","μg/m³"),("PM2.5","PM25","μg/m³"),
                          ("Θόρυβος","N","dBA"),("Πίεση","P","hPa")]:
        a(f"     {lbl:<14}: {v.get(key)} {unit}")
    tf=r['total_f']; cp=r['comfort_pct']
    a(f"\n  2. STUDENT SATISFACTION  (±90 λεπτά | {r['f_start'].strftime('%H:%M')} – {r['f_end'].strftime('%H:%M')})")
    a("-"*65)
    if tf==0: a("     Feedbacks    : 0  →  Δεν υπάρχουν δεδομένα.")
    elif not r['fb_col']: a(f"     Feedbacks    : {tf}  →  Δεν βρέθηκε στήλη feedback.")
    else:
        a(f"     Feedbacks    : {tf}"); a(f"     Satisfaction : {cp:.1f}%"); a("     "+"·"*40)
        for lbl,cnt in sorted(r['category_counts'].items(),key=lambda x:-x[1]):
            a(f"     {lbl:<22}: {cnt:>2}  ({cnt/tf*100:.1f}%)")
    decisions=r['decisions_main']+r['decisions_common']
    has_alert=any(s=='alert' for s,*_ in decisions)
    has_warn =any(s=='warn'  for s,*_ in decisions)
    pt="ΠΕΡΙΟΔΟΣ 1 – Θερμή [Σεπ–20 Νοε]" if r['P1'] else "ΠΕΡΙΟΔΟΣ 2 – Ψυχρή [20 Νοε–Ιαν]"
    a(f"\n  3. DECISION TREE  |  {pt}"); a("="*65)
    out=[(s,p,m) for s,p,m in decisions if s!='ok']
    if not out: a("  ✅  Όλες οι παράμετροι εντός ορίων.")
    else:
        for status,param,msg in out:
            icon={'warn':'⚠️','alert':'🚨','info':'ℹ️'}.get(status,'')
            a(f"  {icon}  {param}\n        → {msg}")
    a("="*65)
    if has_alert:  a("  🚨  ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΑΠΑΙΤΟΥΝΤΑΙ ΑΜΕΣΕΣ ΕΝΕΡΓΕΙΕΣ")
    elif has_warn: a("  ⚠️   ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΧΡΕΙΑΖΕΤΑΙ ΠΑΡΑΚΟΛΟΥΘΗΣΗ")
    else:          a("  ✅  ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΒΕΛΤΙΣΤΕΣ ΣΥΝΘΗΚΕΣ")
    return '\n'.join(lines)

#  PDF EXPORT
def generate_pdf(report_text, v, comfort_pct, category_counts, total_f,
                 target_time, season_id, r=None, sel_date=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as C
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rlc
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.platypus import Table, TableStyle
    except ImportError:
        return None, "pip install reportlab"

    PW, PH = A4
    ML = 18*mm; MR = PW-18*mm; TW = MR-ML; LH = 5.6*mm

    CB   = C.HexColor("#2980B9"); CD   = C.HexColor("#2C3E50")
    CG   = C.HexColor("#7F8C8D"); CR   = C.HexColor("#C0392B")
    CO   = C.HexColor("#D4890A"); CGR  = C.HexColor("#27AE60")
    CBLUE_LIGHT = C.HexColor("#EBF5FB"); CGRAY_LIGHT = C.HexColor("#F2F3F4")
    CRED_LIGHT  = C.HexColor("#FDEDEC"); CGRN_LIGHT  = C.HexColor("#EAFAF1")
    CORG_LIGHT  = C.HexColor("#FEF9E7")

    def ff(n):
        for d in ["/usr/share/fonts/truetype/dejavu","/usr/share/fonts/dejavu","C:\\Windows\\Fonts"]:
            p = os.path.join(d, n)
            if os.path.exists(p): return p
        return None
    sr = ff("DejaVuSans.ttf"); sb = ff("DejaVuSans-Bold.ttf")
    if sr:
        try: pdfmetrics.registerFont(TTFont("Sans", sr))
        except: sr = None
    if sb:
        try: pdfmetrics.registerFont(TTFont("Sans-Bold", sb))
        except: sb = None
    FM  = "Sans"      if sr else "Helvetica"
    FMB = "Sans-Bold" if sb else "Helvetica-Bold"

    buf = io.BytesIO()
    cv  = rlc.Canvas(buf, pagesize=A4)
    pn  = [1]

    def draw_footer():
        cv.setStrokeColor(CG); cv.setLineWidth(0.3)
        cv.line(ML, 13*mm, MR, 13*mm)
        cv.setFont(FM, 7); cv.setFillColor(CG)
        room_str = f"{r['room']} / {r['part']}" if r else ""
        cv.drawString(ML, 9*mm,
            f"Domognostics Pro  |  IAQ Analysis Report  |  {room_str}  |  "
            f"{target_time.strftime('%d/%m/%Y %H:%M')}")
        cv.drawRightString(MR, 9*mm, f"Σελίδα {pn[0]}")

    def draw_page_header(title, sub=None):
        cv.setFont(FMB, 13); cv.setFillColor(CD)
        cv.drawString(ML, PH-20*mm, title)
        if sub:
            cv.setFont(FM, 8.5); cv.setFillColor(CG)
            cv.drawString(ML, PH-27*mm, sub)
        cv.setStrokeColor(CB); cv.setLineWidth(1.2)
        cv.line(ML, PH-29*mm, MR, PH-29*mm)

    def section_title(y, text, color=CB):
        cv.setFont(FMB, 10); cv.setFillColor(color)
        cv.drawString(ML, y, text)
        cv.setStrokeColor(color); cv.setLineWidth(0.5)
        cv.line(ML, y-2*mm, MR, y-2*mm)
        return y - 8*mm

    def embed_image(buf_io, y, width=None, height=None, label=None):
        w = width or TW; h = height or w*0.44
        if label:
            cv.setFont(FMB, 9); cv.setFillColor(CD)
            cv.drawString(ML, y, label); y -= 4*mm
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(buf_io.read()); tmp.close()
        cv.drawImage(tmp.name, ML, y-h, width=w, height=h,
                     preserveAspectRatio=True, mask="auto")
        os.unlink(tmp.name)
        return y - h - 6*mm

    def check_page(y, needed=40*mm, title=None, sub=None):
        if y < needed:
            cv.showPage(); pn[0] += 1; draw_footer()
            if title: draw_page_header(title, sub)
            return PH - (34*mm if title else 20*mm)
        return y

    P1 = "ΠΕΡΙΟΔΟΣ 1" in season_id
    room_str   = f"{r['room']} / {r['part']}" if r else "N/A"
    period_str = "Περίοδος 1 — Θερμή/Μεταβατική (Σεπ – 20 Νοε)" if P1                  else "Περίοδος 2 — Ψυχρή (20 Νοε – Ιαν)"
    decisions  = (r.get("decisions_main",[]) + r.get("decisions_common",[])) if r else []
    has_alert  = any(s=="alert" for s,*_ in decisions)
    has_warn   = any(s=="warn"  for s,*_ in decisions)

    # PAGE 1 
    cv.setFillColor(CD)
    cv.rect(0, PH-40*mm, PW, 40*mm, fill=1, stroke=0)
    cv.setFillColor(C.white); cv.setFont(FMB, 22)
    cv.drawCentredString(PW/2, PH-17*mm, "DOMOGNOSTICS PRO")
    cv.setFont(FM, 11)
    cv.drawCentredString(PW/2, PH-26*mm, "Indoor Air Quality Analysis Report")
    cv.setFont(FM, 8.5); cv.setFillColor(C.HexColor("#AED6F1"))
    cv.drawCentredString(PW/2, PH-34*mm,
        "Objective Sensor Measurements  ·  Subjective Student Perception  ·  Decision Support")

    cv.setStrokeColor(CB); cv.setLineWidth(3)
    cv.line(ML, PH-43*mm, MR, PH-43*mm)

    y_c = PH - 53*mm

    # Info box
    cv.setFillColor(CBLUE_LIGHT)
    cv.roundRect(ML, y_c-58*mm, TW, 58*mm, 4*mm, fill=1, stroke=0)
    info_items = [
        ("Ημερομηνία & Ώρα:", target_time.strftime("%d/%m/%Y  %H:%M")),
        ("Αίθουσα / Τμήμα:", room_str),
        ("Κλιματική Περίοδος:", period_str),
        ("Sensor CSV:", os.path.basename(r["s_path"]) if r and r.get("s_path") else "N/A"),
        ("Feedback CSV:", os.path.basename(r["f_path"]) if r and r.get("f_path") else "N/A"),
        ("Feedbacks / Satisfaction:",
         f"{total_f}  |  {comfort_pct:.1f}%" if comfort_pct is not None else str(total_f)),
    ]
    yi = y_c - 9*mm
    for lbl, val in info_items:
        cv.setFont(FMB, 9); cv.setFillColor(CG); cv.drawString(ML+5*mm, yi, lbl)
        cv.setFont(FM, 9);  cv.setFillColor(CB); cv.drawString(ML+68*mm, yi, val)
        yi -= 8*mm
    y_c -= 64*mm

    # Status box
    if has_alert:
        bc, tc, st = CRED_LIGHT,  CR,  "!! ΑΠΑΙΤΟΥΝΤΑΙ ΑΜΕΣΕΣ ΕΝΕΡΓΕΙΕΣ"
    elif has_warn:
        bc, tc, st = CORG_LIGHT,  CO,  "!  ΧΡΕΙΑΖΕΤΑΙ ΠΑΡΑΚΟΛΟΥΘΗΣΗ"
    else:
        bc, tc, st = CGRN_LIGHT,  CGR, "OK ΒΕΛΤΙΣΤΕΣ ΣΥΝΘΗΚΕΣ"
    y_c -= 6*mm
    cv.setFillColor(bc); cv.roundRect(ML, y_c-14*mm, TW, 14*mm, 3*mm, fill=1, stroke=0)
    cv.setFont(FMB, 11); cv.setFillColor(tc)
    cv.drawCentredString(PW/2, y_c-9*mm, f"ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ:  {st}")
    y_c -= 22*mm

    # Metrics grid
    metrics = [
        ("Θερμοκρασία", v.get("T"), "°C",    (23,27) if P1 else (20,24)),
        ("Υγρασία",     v.get("H"), "%",      (40,60) if P1 else (30,50)),
        ("CO2",         v.get("C"), "ppm",    (0,1000) if P1 else (0,1200)),
        ("VOC",         v.get("VOC"),"ppb",   (0,100)),
        ("PM1",         v.get("PM1"),"μg/m3", (0,10)),
        ("PM2.5",       v.get("PM25"),"μg/m3",(0,12)),
        ("Θόρυβος",     v.get("N"), "dBA",    (0,35)),
        ("Πίεση",       v.get("P"), "hPa",    (980,1050)),
    ]
    cw = TW/4; ch = 18*mm
    for i, (lbl, val, unit, lims) in enumerate(metrics):
        ci = i%4; ri = i//4; x = ML+ci*cw; yy = y_c - ri*ch
        ir = (val not in (None,"N/A")) and lims[0]<=val<=lims[1]
        oor= (val not in (None,"N/A")) and not ir
        bg = CGRN_LIGHT if ir else (CRED_LIGHT if oor else CGRAY_LIGHT)
        cv.setFillColor(bg)
        cv.roundRect(x+1*mm, yy-ch+1*mm, cw-2*mm, ch-2*mm, 2*mm, fill=1, stroke=0)
        cv.setFont(FM, 7.5); cv.setFillColor(CG)
        cv.drawCentredString(x+cw/2, yy-5*mm, lbl)
        vs = f"{val} {unit}" if val not in (None,"N/A") else "N/A"
        cv.setFont(FMB, 9.5)
        cv.setFillColor(CGR if ir else (CR if oor else CG))
        cv.drawCentredString(x+cw/2, yy-12*mm, vs)

    draw_footer()

    # PAGE 2 — SENSOR TABLE + SATISFACTION + SUGGESTIONS
    cv.showPage(); pn[0] = 2; draw_footer()
    draw_page_header("Αναλυτικά Δεδομένα & Προτάσεις",
                     f"{room_str}  |  {target_time.strftime('%d/%m/%Y %H:%M')}  |  {period_str}")
    y2 = PH - 35*mm

    # Sensor table
    y2 = section_title(y2, "1.  Δεδομένα Αισθητήρα  (±90 λεπτά)")
    PLIMITS = {
        "Θερμοκρασία":((23,27) if P1 else (20,24),"°C"),
        "Υγρασία":    ((40,60) if P1 else (30,50),"%"),
        "CO2":        ((0,1000) if P1 else (0,1200),"ppm"),
        "VOC":        ((0,100),"ppb"), "PM1":((0,10),"μg/m3"),
        "PM2.5":      ((0,12),"μg/m3"), "Θόρυβος":((0,35),"dBA"),
        "Πίεση":      ((980,1050),"hPa"),
    }
    PKEYS = {"Θερμοκρασία":"T","Υγρασία":"H","CO2":"C","VOC":"VOC",
             "PM1":"PM1","PM2.5":"PM25","Θόρυβος":"N","Πίεση":"P"}
    td = [["Παράμετρος","Τιμή","Μονάδα","Βέλτιστο","Κατάσταση"]]
    for pname,(lims,unit) in PLIMITS.items():
        val = v.get(PKEYS[pname])
        vs  = str(val) if val not in (None,"N/A") else "N/A"
        rs  = f"{lims[0]} – {lims[1]}"
        if val not in (None,"N/A"):
            st2 = "Εντος" if lims[0]<=val<=lims[1] else "Εκτος"
        else: st2 = "N/A"
        td.append([pname, vs, unit, rs, st2])
    tbl1 = Table(td, colWidths=[40*mm,22*mm,18*mm,35*mm,35*mm])
    status_styles = []
    for i in range(1, len(td)):
        sc = CGR if td[i][4]=="Εντος" else (CR if td[i][4]=="Εκτος" else CG)
        status_styles += [("TEXTCOLOR",(4,i),(4,i),sc),("FONTNAME",(4,i),(4,i),FMB)]
    tbl1.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),CD),("TEXTCOLOR",(0,0),(-1,0),C.white),
        ("FONTNAME",(0,0),(-1,0),FMB),("FONTSIZE",(0,0),(-1,0),8.5),
        ("FONTNAME",(0,1),(-1,-1),FM),("FONTSIZE",(0,1),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C.HexColor("#F8F9FA"),C.white]),
        ("GRID",(0,0),(-1,-1),0.3,C.HexColor("#BDC3C7")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("ALIGN",(0,0),(0,-1),"LEFT"),
        ("LEFTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]+status_styles))
    tbl1.wrapOn(cv, TW, PH); h1=tbl1._height
    tbl1.drawOn(cv, ML, y2-h1); y2 -= h1+8*mm

    # Satisfaction
    y2 = check_page(y2, 50*mm, "Αναλυτικά Δεδομένα & Προτάσεις (συνέχεια)")
    y2 = section_title(y2, "2.  Student Satisfaction Index  (±90 λεπτά)")
    if total_f == 0:
        cv.setFont(FM,9); cv.setFillColor(CG)
        cv.drawString(ML, y2, "Δεν υπάρχουν feedbacks στο χρονικό παράθυρο."); y2-=LH*2
    else:
        sat = comfort_pct or 0
        bw  = 110*mm
        cv.setFillColor(C.HexColor("#ECF0F1"))
        cv.roundRect(ML, y2-8*mm, bw, 8*mm, 2*mm, fill=1, stroke=0)
        fc = CGR if sat>=65 else (CO if sat>=40 else CR)
        cv.setFillColor(fc)
        cv.roundRect(ML, y2-8*mm, bw*(sat/100), 8*mm, 2*mm, fill=1, stroke=0)
        cv.setFont(FMB,9); cv.setFillColor(CD)
        cv.drawString(ML+bw+4*mm, y2-5.5*mm,
                      f"Satisfaction Index: {sat:.1f}%  |  N={total_f}")
        y2 -= 13*mm
        if category_counts:
            fbd=[["Κατηγορία","Πλήθος","%","Βάρος","Τύπος"]]
            for lbl,cnt in sorted(category_counts.items(),key=lambda x:-x[1]):
                w=FEEDBACK_WEIGHTS.get(lbl.lower(),0.5)
                typ="Θετικο" if w==1.0 else ("Ουδετερο" if w==0.5 else "Αρνητικο")
                fbd.append([lbl,str(cnt),f"{cnt/total_f*100:.1f}%",f"{w:.1f}",typ])
            tbl2=Table(fbd,colWidths=[55*mm,22*mm,22*mm,22*mm,29*mm])
            typ_styles=[]
            for i in range(1,len(fbd)):
                tc2=CGR if fbd[i][4]=="Θετικο" else (CO if fbd[i][4]=="Ουδετερο" else CR)
                typ_styles+=[("TEXTCOLOR",(4,i),(4,i),tc2),("FONTNAME",(4,i),(4,i),FMB)]
            tbl2.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),CD),("TEXTCOLOR",(0,0),(-1,0),C.white),
                ("FONTNAME",(0,0),(-1,0),FMB),("FONTSIZE",(0,0),(-1,0),8.5),
                ("FONTNAME",(0,1),(-1,-1),FM),("FONTSIZE",(0,1),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[C.HexColor("#F8F9FA"),C.white]),
                ("GRID",(0,0),(-1,-1),0.3,C.HexColor("#BDC3C7")),
                ("ALIGN",(1,0),(-1,-1),"CENTER"),
                ("LEFTPADDING",(0,0),(-1,-1),4),
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ]+typ_styles))
            tbl2.wrapOn(cv,TW,PH); h2=tbl2._height
            y2=check_page(y2, h2+10*mm, "Αναλυτικά Δεδομένα & Προτάσεις (συνέχεια)")
            tbl2.drawOn(cv,ML,y2-h2); y2-=h2+8*mm

    # Suggestions
    y2=check_page(y2,50*mm,"Προτάσεις & Αξιολόγηση",
                  f"{room_str}  |  {'Περίοδος 1' if P1 else 'Περίοδος 2'}")
    y2=section_title(y2,f"3.  Προτάσεις  |  {'Περίοδος 1 — Θερμή' if P1 else 'Περίοδος 2 — Ψυχρή'}")
    ICONS_P={"ok":"[OK]","warn":"[!] ","alert":"[!!]","info":"[i] "}
    BG_P={"ok":CGRN_LIGHT,"warn":CORG_LIGHT,"alert":CRED_LIGHT,"info":CBLUE_LIGHT}
    SC_P={"ok":CGR,"warn":CO,"alert":CR,"info":CB}
    for status,param,msg in decisions:
        bh = LH*2+3*mm
        y2=check_page(y2,bh+4*mm,"Προτάσεις (συνέχεια)")
        cv.setFillColor(BG_P.get(status,C.white))
        cv.roundRect(ML,y2-bh,TW,bh,1.5*mm,fill=1,stroke=0)
        cv.setFont(FMB,8.5); cv.setFillColor(SC_P.get(status,CD))
        cv.drawString(ML+3*mm, y2-5.5*mm, f"{ICONS_P.get(status,'')}  {param}")
        cv.setFont(FM,8); cv.setFillColor(CD)
        cv.drawString(ML+8*mm, y2-5.5*mm-LH, f"-> {msg}")
        y2 -= bh+2*mm

    # PAGE 3 — VISUAL ANALYSIS
    cv.showPage(); pn[0]+=1; draw_footer()
    draw_page_header("Visual Analysis",
                     "Sensor Data · Satisfaction · Correlation Heatmap · Feedback Distribution")
    y3=PH-35*mm

    try:
        c1=chart_dual_axis(v,comfort_pct,target_time,season_id)
        y3=embed_image(c1,y3,TW,TW*0.42,"Sensor Data & Satisfaction Index")
    except Exception: pass

    y3=check_page(y3,80*mm,"Visual Analysis (συνέχεια)")

    # Heatmap + Pie side by side
    half=TW/2-3*mm
    y_side=y3
    try:
        c2=chart_heatmap(v,category_counts,total_f)
        cv.setFont(FMB,9); cv.setFillColor(CD)
        cv.drawString(ML,y_side,"Correlation Heatmap")
        tmp2=tempfile.NamedTemporaryFile(suffix=".png",delete=False)
        tmp2.write(c2.read()); tmp2.close()
        h2s=half*0.65
        cv.drawImage(tmp2.name,ML,y_side-4*mm-h2s,width=half,height=h2s,
                     preserveAspectRatio=True,mask="auto")
        os.unlink(tmp2.name)
    except Exception: pass

    if total_f>0 and category_counts:
        try:
            c3=chart_pie(category_counts,total_f,comfort_pct or 0)
            if c3:
                cv.setFont(FMB,9); cv.setFillColor(CD)
                cv.drawString(ML+half+6*mm,y_side,"Κατανομη Feedback")
                tmp3=tempfile.NamedTemporaryFile(suffix=".png",delete=False)
                tmp3.write(c3.read()); tmp3.close()
                h3s=half*0.72
                cv.drawImage(tmp3.name,ML+half+6*mm,y_side-4*mm-h3s,
                             width=half,height=h3s,preserveAspectRatio=True,mask="auto")
                os.unlink(tmp3.name)
        except Exception: pass
    y3 -= half*0.75+10*mm

    # 
    # PAGE 4  MONTHLY STATS
    if r and r.get("s_path") and r.get("f_path"):
        try:
            mstats=compute_monthly_stats(r["s_path"],r["f_path"])
            if "sensor_df" in mstats:
                cv.showPage(); pn[0]+=1; draw_footer()
                mn=MONTH_FEEDBACK_NAME.get(target_time.strftime("%m"),"")+' '+str(target_time.year)
                draw_page_header("Μηνιαία Στατιστική Ανάλυση",f"{room_str}  |  {mn}")
                y4=PH-35*mm
                y4=section_title(y4,"Στατιστικα Αισθητηρα — Μηνιαια Συνοψη")
                sdf=mstats["sensor_df"]
                hdr2=["Παράμετρος","Μέσος","Τυπ.Απόκλ.","Ελάχ.","Μέγ.","Μεσαία","N"]
                td2=[hdr2]
                for _,row in sdf.iterrows():
                    td2.append([str(row["Παράμετρος"]),str(row["Μέσος Όρος"]),
                                str(row["Τυπ. Απόκλιση"]),str(row["Ελάχιστο"]),
                                str(row["Μέγιστο"]),str(row["Μεσαία Τιμή"]),
                                str(row["N Μετρήσεις"])])
                tbl_m=Table(td2,colWidths=[48*mm,20*mm,20*mm,18*mm,18*mm,18*mm,18*mm])
                tbl_m.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),CD),("TEXTCOLOR",(0,0),(-1,0),C.white),
                    ("FONTNAME",(0,0),(-1,0),FMB),("FONTSIZE",(0,0),(-1,0),7.5),
                    ("FONTNAME",(0,1),(-1,-1),FM),("FONTSIZE",(0,1),(-1,-1),7.5),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[C.HexColor("#F8F9FA"),C.white]),
                    ("GRID",(0,0),(-1,-1),0.3,C.HexColor("#BDC3C7")),
                    ("ALIGN",(1,0),(-1,-1),"CENTER"),
                    ("LEFTPADDING",(0,0),(-1,-1),3),
                    ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
                ]))
                tbl_m.wrapOn(cv,TW,PH); tm_h=tbl_m._height
                tbl_m.drawOn(cv,ML,y4-tm_h); y4-=tm_h+8*mm
                try:
                    bp=chart_monthly_boxplot(r["s_path"],P1)
                    if bp:
                        y4=section_title(y4,"Κατανομη Τιμων — Box Plot")
                        y4=embed_image(bp,y4,TW,TW*0.38)
                except Exception: pass
                if "monthly_satisfaction" in mstats and mstats["monthly_satisfaction"] is not None:
                    ms=mstats["monthly_satisfaction"]; tf_m=mstats.get("total_feedbacks",0)
                    y4=check_page(y4,40*mm,"Μηνιαία Στατιστική (συνέχεια)")
                    y4=section_title(y4,"Μηνιαιο Satisfaction Index")
                    bw2=110*mm
                    cv.setFillColor(C.HexColor("#ECF0F1"))
                    cv.roundRect(ML,y4-8*mm,bw2,8*mm,2*mm,fill=1,stroke=0)
                    fc2=CGR if ms>=65 else (CO if ms>=40 else CR)
                    cv.setFillColor(fc2)
                    cv.roundRect(ML,y4-8*mm,bw2*(ms/100),8*mm,2*mm,fill=1,stroke=0)
                    cv.setFont(FMB,9); cv.setFillColor(CD)
                    cv.drawString(ML+bw2+4*mm,y4-5.5*mm,f"{ms:.1f}%  |  N={tf_m} feedbacks")
                    y4-=14*mm
        except Exception: pass

    
    # METHODOLOGY
    cv.showPage(); pn[0]+=1; draw_footer()
    draw_page_header("Μεθοδολογία & Πρότυπα Αναφοράς",
                     "Βάσεις αξιολόγησης · Satisfaction Index · Κλιματικές Περίοδοι")
    ym=PH-35*mm
    ym=section_title(ym,"Κλιματικές Περίοδοι")
    for pt,pd2 in [("Περίοδος 1 — Θερμή/Μεταβατική",
                    "1 Σεπτεμβρίου – 20 Νοεμβρίου. Αερισμός: φυσικός (άνοιγμα παραθύρων)."),
                   ("Περίοδος 2 — Ψυχρή",
                    "21 Νοεμβρίου – 31 Ιανουαρίου. Αερισμός: μηχανικός μόνο (HVAC/A/C).")]:
        cv.setFont(FMB,8.5); cv.setFillColor(CB); cv.drawString(ML,ym,pt); ym-=LH
        cv.setFont(FM,8); cv.setFillColor(CD); cv.drawString(ML+4*mm,ym,pd2); ym-=LH+2*mm
    ym=section_title(ym,"Όρια Παραμέτρων (ASHRAE 55 / WHO / EN 16798)")
    lim_td=[
        ["Παράμετρος","Περίοδος 1","Περίοδος 2","Πρότυπο"],
        ["Θερμοκρασία","23–27 °C","20–24 °C","ASHRAE 55"],
        ["Υγρασία","40–60 %","30–50 %","ASHRAE 55"],
        ["CO2","< 1000 ppm","< 1200 ppm","EN 16798"],
        ["VOC","< 300 μg/m3","< 300 μg/m3","WHO IAQ"],
        ["PM2.5","< 12 μg/m3","< 12 μg/m3","WHO 2021"],
        ["PM1","< 10 μg/m3","< 10 μg/m3","WHO IAQ"],
        ["Θόρυβος","< 35 dBA","< 35 dBA","WHO"],
        ["Πίεση","980–1050 hPa","980–1050 hPa","Ατμοσφαιρική"],
    ]
    tbl_lim=Table(lim_td,colWidths=[45*mm,38*mm,38*mm,29*mm])
    tbl_lim.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),CD),("TEXTCOLOR",(0,0),(-1,0),C.white),
        ("FONTNAME",(0,0),(-1,0),FMB),("FONTSIZE",(0,0),(-1,0),8),
        ("FONTNAME",(0,1),(-1,-1),FM),("FONTSIZE",(0,1),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[C.HexColor("#F8F9FA"),C.white]),
        ("GRID",(0,0),(-1,-1),0.3,C.HexColor("#BDC3C7")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("LEFTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    tbl_lim.wrapOn(cv,TW,PH); lh2=tbl_lim._height
    tbl_lim.drawOn(cv,ML,ym-lh2); ym-=lh2+8*mm
    ym=section_title(ym,"Weighted Satisfaction Index — Φόρμουλα")
    for fl in [
        "Satisfaction Index (%) = ( Sum(βάρος × πλήθος) ) / N × 100",
        "  Comfortable / Pleasant  ->  βάρος 1.0  (πλήρης άνεση)",
        "  Neutral / Noticeable    ->  βάρος 0.5  (μέτρια άνεση)",
        "  Too Hot/Cold/Dry/Humid / Irritating / Unpleasant  ->  βάρος 0.0  (δυσφορία)",
        "  Ερμηνεία:  >= 65% = Καλό  |  40–64% = Μέτριο  |  < 40% = Χαμηλό",
    ]:
        cv.setFont(FMB if "Satisfaction Index" in fl else FM, 8.5 if "Satisfaction Index" in fl else 8)
        cv.setFillColor(CD); cv.drawString(ML,ym,fl); ym-=LH
    ym-=6*mm
    cv.setFillColor(CGRAY_LIGHT)
    cv.roundRect(ML,ym-12*mm,TW,12*mm,2*mm,fill=1,stroke=0)
    cv.setFont(FM,7.5); cv.setFillColor(CG)
    cv.drawCentredString(PW/2, ym-7*mm,
        f"Αναφορά: {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
        "Domognostics Pro  |  Πρότυπα: ASHRAE 55, WHO, EN 16798")
    cv.save(); buf.seek(0)
    return buf, None



#  COMPARISON ACROSS PARTS  (DRAKOS / TASOS)
def load_part_data(base_path, room, part, sel_date, hour, minute):
    """Φορτώνει sensor + feedback δεδομένα για ένα συγκεκριμένο τμήμα."""
    month_num   = sel_date.strftime('%m')
    target_time = pd.Timestamp(year=sel_date.year, month=sel_date.month, day=sel_date.day,
                               hour=hour, minute=minute)
    s_start = target_time - pd.Timedelta(minutes=90)
    s_end   = target_time + pd.Timedelta(minutes=90)
    f_start = target_time - pd.Timedelta(minutes=90)
    f_end   = target_time + pd.Timedelta(minutes=90)

    s_path = find_sensor_path(base_path, room, part, month_num)
    f_path = find_feedback_path(base_path, room, part, month_num)

    v = {k: "N/A" for k in ['T','H','C','VOC','PM1','PM25','N','P']}
    comfort_pct = None; total_f = 0; found_sensor = False; found_feedback = False

    if s_path:
        df_s = read_csv_path(s_path)
        if df_s is not None:
            df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
            snap = df_s[(df_s['Timestamp'] >= s_start) & (df_s['Timestamp'] <= s_end)]
            if not snap.empty:
                found_sensor = True
                m_col = next((c for c in snap.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                             next((c for c in snap.columns if 'meas' in c.lower()), None))
                v_col = next((c for c in snap.columns if c.lower() == 'value'),
                             next((c for c in snap.columns if 'valu' in c.lower()), None))
                if m_col and v_col:
                    def gv(names):
                        mask = snap[m_col].astype(str).str.strip().str.lower().isin([n.lower() for n in names])
                        vals = pd.to_numeric(snap.loc[mask, v_col], errors='coerce').dropna()
                        return round(vals.mean(), 1) if not vals.empty else "N/A"
                    v = {
                        'T':   gv(['Temperature']),
                        'H':   gv(['Humidity']),
                        'C':   gv(['Carbon Dioxide','CO2']),
                        'VOC': gv(['VOC','Volatile Organic Compounds']),
                        'PM1': gv(['PM1','PM 1','PM1.0','pm1.0']),
                        'PM25':gv(['PM2.5','PM 2.5','pm2.5']),
                        'N':   gv(['Noise']),
                        'P':   gv(['Pressure']),
                    }

    if f_path:
        df_f = read_csv_path(f_path)
        if df_f is not None:
            df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')
            wf = df_f[(df_f['Timestamp'] >= f_start) & (df_f['Timestamp'] <= f_end)]
            total_f = len(wf)
            if total_f > 0:
                found_feedback = True
                fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in wf.columns else None
                if not fb_col:
                    fb_col = next((c for c in wf.columns
                                   if any(k in c.lower() for k in ('temp','feed','comfort'))), None)
                if fb_col:
                    cats = wf[fb_col].astype(str).str.strip().value_counts().to_dict()
                    ws   = sum(FEEDBACK_WEIGHTS.get(l.lower(), 0.0)*cnt for l,cnt in cats.items())
                    comfort_pct = round((ws / total_f) * 100, 1)

    return {
        'part': part, 'v': v, 'comfort_pct': comfort_pct,
        'total_f': total_f, 'found_sensor': found_sensor, 'found_feedback': found_feedback,
        's_path': s_path, 'f_path': f_path,
    }


def chart_comparison_bars(parts_data, param_key, param_label, unit, good_range=None):
    """
    Bar chart σύγκρισης μιας παραμέτρου μεταξύ όλων των τμημάτων.
    good_range = (min, max) για να χρωματίσουμε τις μπάρες.
    """
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    labels = [d['part'].replace('_',' ') for d in parts_data]
    values = []
    for d in parts_data:
        val = d['v'].get(param_key)
        values.append(val if val not in (None, "N/A") else 0)

    # Χρώμα μπάρας ανάλογα με τα όρια των παραμέτρων
    bar_colors = []
    for d in parts_data:
        val = d['v'].get(param_key)
        if val in (None, "N/A"):
            bar_colors.append('#95A5A6')
        elif good_range and good_range[0] <= val <= good_range[1]:
            bar_colors.append('#27AE60')
        elif good_range:
            bar_colors.append('#E74C3C')
        else:
            bar_colors.append('#3498DB')

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#FFFFFF')
    bars = ax.bar(labels, values, color=bar_colors, alpha=0.85, width=0.5,
                  edgecolor='white', linewidth=0.8)
    for bar, d in zip(bars, parts_data):
        val = d['v'].get(param_key)
        if val not in (None, "N/A"):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.02,
                    f'{val}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2C3E50')

    # Γραμμή ορίων
    if good_range:
        ax.axhline(y=good_range[0], color='#F39C12', linestyle=':', linewidth=1.2, alpha=0.8)
        ax.axhline(y=good_range[1], color='#E74C3C', linestyle=':', linewidth=1.2, alpha=0.8)
        ymax = max(max(values)*1.3, good_range[1]*1.2) if values else good_range[1]*1.2
    else:
        ymax = max(values)*1.3 if values else 10

    ax.set_ylim(0, ymax)
    ax.set_ylabel(f'{param_label} ({unit})', fontsize=9, color='#2C3E50')
    ax.set_title(f'{param_label} — Σύγκριση Τμημάτων', fontsize=10, pad=8,
                 color='#2C3E50', fontweight='bold')
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Legend ορίων
    if good_range:
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0],[0], color='#F39C12', linestyle=':', label=f'Κάτω όριο ({good_range[0]} {unit})'),
            Line2D([0],[0], color='#E74C3C', linestyle=':', label=f'Άνω όριο ({good_range[1]} {unit})'),
        ]
        ax.legend(handles=legend_els, fontsize=7, loc='upper right', framealpha=0.8)

    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


def chart_satisfaction_comparison(parts_data):
    """Bar chart σύγκρισης Satisfaction Index μεταξύ τμημάτων."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    labels = [d['part'].replace('_',' ') for d in parts_data]
    values = [d['comfort_pct'] if d['comfort_pct'] is not None else 0 for d in parts_data]
    colors = ['#27AE60' if v>=65 else ('#F39C12' if v>=40 else '#E74C3C') for v in values]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#FFFFFF')
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5,
                  edgecolor='white', linewidth=0.8)
    for bar, d, v in zip(bars, parts_data, values):
        label = f"{v:.1f}%\n(N={d['total_f']})" if d['total_f']>0 else "N/A"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                label, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#2C3E50')

    ax.axhline(y=65, color='#27AE60', linestyle='--', linewidth=1.2, alpha=0.7, label='Καλό (≥65%)')
    ax.axhline(y=40, color='#F39C12', linestyle='--', linewidth=1.2, alpha=0.7, label='Μέτριο (≥40%)')
    ax.set_ylim(0, 120)
    ax.set_ylabel('Satisfaction Index (%)', fontsize=9, color='#2C3E50')
    ax.set_title('Student Satisfaction Index — Σύγκριση Τμημάτων', fontsize=10, pad=8,
                 color='#2C3E50', fontweight='bold')
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


def chart_radar_comparison(parts_data, P1):
    """
    Radar/Spider chart — κανονικοποιημένες παράμετροι για κάθε τμήμα.
    Κάθε παράμετρος κανονικοποιείται 0–1 (1=βέλτιστο, 0=χειρότερο).
    """
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    # Ορισμός βέλτιστων εύρων ανά περίοδο
    RANGES = {
        'T':    (23,27) if P1 else (20,24),
        'H':    (40,60) if P1 else (30,50),
        'C':    (400,1000) if P1 else (400,1200),
        'VOC':  (0,100),
        'PM25': (0,12),
        'N':    (0,35),
    }
    LABELS = ['Θερμοκρασία','Υγρασία','CO2','VOC','PM2.5','Θόρυβος']
    KEYS   = ['T','H','C','VOC','PM25','N']

    def normalize(key, val):
        if val in (None,"N/A"): return 0.5
        lo, hi = RANGES[key]
        # Για παραμέτρους όπου χαμηλότερο = είναι καλύτερο (CO2, VOC, PM2.5, Noise)
        if key in ('C','VOC','PM25','N'):
            if val <= lo: return 1.0
            if val >= hi*1.5: return 0.0
            return max(0, 1 - (val - lo) / (hi*1.5 - lo))
        else:  # T και H = καλύτερο = εντός εύρους
            mid = (lo + hi) / 2
            span = (hi - lo) / 2
            return max(0, 1 - abs(val - mid) / (span * 2))

    N_params = len(KEYS)
    angles = [n / float(N_params) * 2 * np.pi for n in range(N_params)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#F0F3F4')

    PART_COLORS = ['#2980B9','#E74C3C','#27AE60','#F39C12','#9B59B6','#1ABC9C']

    for i, d in enumerate(parts_data):
        vals = [normalize(k, d['v'].get(k)) for k in KEYS]
        vals += vals[:1]
        color = PART_COLORS[i % len(PART_COLORS)]
        ax.plot(angles, vals, 'o-', linewidth=2, color=color,
                label=d['part'].replace('_',' '))
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=8.5, color='#2C3E50')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], fontsize=6.5, color='#7F8C8D')
    ax.grid(color='#BDC3C7', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_title('IAQ Score — Σύγκριση Τμημάτων\n(1.0 = Βέλτιστο)',
                 fontsize=10, pad=18, color='#2C3E50', fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8, framealpha=0.9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


#  DAY/HOUR HEATMAP
def chart_day_hour_heatmap(s_path, f_path, param_key, param_names,
                           param_label, unit, limits, month_label):
    """
    Heatmap: Άξονας X = ημέρα μήνα, Άξονας Y = ώρα ημέρας.
    Χρώμα = τιμή παραμέτρου ή IAQ score.
    """
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    df_s = read_csv_path(s_path)
    if df_s is None: return None, "Αδυναμία ανάγνωσης CSV."

    df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
    m_col = next((c for c in df_s.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                 next((c for c in df_s.columns if 'meas' in c.lower()), None))
    v_col = next((c for c in df_s.columns if c.lower()=='value'),
                 next((c for c in df_s.columns if 'valu' in c.lower()), None))
    if not m_col or not v_col:
        return None, "Δεν βρέθηκαν στήλες."

    mask = df_s[m_col].astype(str).str.strip().str.lower().isin(
        [n.lower() for n in param_names])
    df_p = df_s[mask].copy()
    df_p[v_col] = pd.to_numeric(df_p[v_col], errors='coerce')
    df_p = df_p.dropna(subset=[v_col,'Timestamp'])
    df_p['day']  = df_p['Timestamp'].dt.day
    df_p['hour'] = df_p['Timestamp'].dt.hour

    if df_p.empty:
        return None, f"Δεν βρέθηκαν δεδομένα για {param_label}."

    # rows=ώρα, cols=ημέρα
    pivot = df_p.groupby(['hour','day'])[v_col].mean().unstack(fill_value=np.nan)

    # Αν υπάρχει satisfaction από feedback προσθέτουμε ως overlay
    sat_pivot = None
    if f_path:
        df_f = read_csv_path(f_path)
        if df_f is not None:
            df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')
            fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in df_f.columns else None
            if not fb_col:
                fb_col = next((c for c in df_f.columns
                               if any(k in c.lower() for k in ('temp','feed','comfort'))),None)
            if fb_col:
                df_f['weight'] = df_f[fb_col].astype(str).str.strip().str.lower().map(
                    lambda x: FEEDBACK_WEIGHTS.get(x, np.nan))
                df_f['day']  = df_f['Timestamp'].dt.day
                df_f['hour'] = df_f['Timestamp'].dt.hour
                sat_pivot = df_f.groupby(['hour','day'])['weight'].mean().unstack(fill_value=np.nan)

    fig_h = max(5, len(pivot.index) * 0.35)
    fig_w = max(10, len(pivot.columns) * 0.55)
    fig, ax = plt.subplots(figsize=(min(fig_w,16), min(fig_h,9)))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#1a1a2e')

    # Χρωματική κλίμακα — πράσινο=καλό, κόκκινο=κακό
    if limits:
        lo, hi = limits
        # Normalize: εντός ορίων = 0.5 (κίτρινο→πράσινο), εκτός = κόκκινο
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
    else:
        cmap = plt.cm.YlOrRd_r

    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                   interpolation='nearest',
                   vmin=limits[0]*0.7 if limits else None,
                   vmax=limits[1]*1.3 if limits else None)

    # Axes labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(d) for d in pivot.columns], fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{h:02d}:00' for h in pivot.index], fontsize=7.5)
    ax.set_xlabel('Ημέρα Μήνα', fontsize=9, color='white')
    ax.set_ylabel('Ώρα', fontsize=9, color='white')
    ax.tick_params(colors='white')

    # Τιμές μέσα στα κελιά 
    if len(pivot.columns) <= 31 and len(pivot.index) <= 24:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i,j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                            fontsize=6, color='black', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f'{param_label} ({unit})', fontsize=8, color='#2C3E50')
    cbar.ax.tick_params(labelsize=7)

    if limits:
        cbar.ax.axhline(y=limits[0], color='white', linewidth=1.5, linestyle='--')
        cbar.ax.axhline(y=limits[1], color='white', linewidth=1.5, linestyle='--')

    ax.set_title(
        f'Heatmap {param_label}  |  {month_label}  '
        f'(Βέλτιστο: {limits[0]}–{limits[1]} {unit})' if limits
        else f'Heatmap {param_label}  |  {month_label}',
        fontsize=10, pad=10, color='#2C3E50', fontweight='bold')

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0)
    return buf, None


def chart_satisfaction_heatmap(f_path, month_label):
    """Heatmap satisfaction: ημέρα × ώρα."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    df_f = read_csv_path(f_path)
    if df_f is None: return None

    df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')
    fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in df_f.columns else None
    if not fb_col:
        fb_col = next((c for c in df_f.columns
                       if any(k in c.lower() for k in ('temp','feed','comfort'))),None)
    if not fb_col: return None

    df_f['weight'] = df_f[fb_col].astype(str).str.strip().str.lower().map(
        lambda x: FEEDBACK_WEIGHTS.get(x, np.nan)) * 100
    df_f['day']  = df_f['Timestamp'].dt.day
    df_f['hour'] = df_f['Timestamp'].dt.hour
    df_f = df_f.dropna(subset=['weight','day','hour'])

    if df_f.empty: return None

    pivot = df_f.groupby(['hour','day'])['weight'].mean().unstack(fill_value=np.nan)

    fig_h = max(4, len(pivot.index)*0.4)
    fig_w = max(8, len(pivot.columns)*0.5)
    fig, ax = plt.subplots(figsize=(min(fig_w,16), min(fig_h,8)))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#1a1a2e')

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   interpolation='nearest', vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(d) for d in pivot.columns], fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{h:02d}:00' for h in pivot.index], fontsize=7.5)
    ax.set_xlabel('Ημέρα Μήνα', fontsize=9, color='white')
    ax.set_ylabel('Ώρα', fontsize=9, color='white')
    ax.tick_params(colors='white')

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i,j]
            if not np.isnan(val):
                tc = 'black' if 30<val<70 else 'white'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=6.5, color=tc, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Satisfaction (%)', fontsize=8, color='#2C3E50')
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f'Heatmap Student Satisfaction  |  {month_label}',
                 fontsize=10, pad=10, color='#2C3E50', fontweight='bold')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0)
    return buf


#  STREAMLIT UI
st.set_page_config(page_title="Domognostics Pro", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:800;color:#2980B9;margin-bottom:0}
.sub-title{font-size:1rem;color:#7F8C8D;margin-top:0}
.decision-row{padding:10px 14px;border-radius:6px;margin-bottom:6px;font-size:.92rem}
.d-ok   {background:#EAFAF1;border-left:4px solid #27AE60}
.d-ok   strong{color:#1A5C35 !important}
.d-ok   span{color:#2C6E3F !important}
.d-warn {background:#FEF9E7;border-left:4px solid #D4890A}
.d-warn strong{color:#7D4E00 !important}
.d-warn span{color:#6B4200 !important}
.d-alert{background:#FDEDEC;border-left:4px solid #C0392B}
.d-alert strong{color:#7B1A1A !important}
.d-alert span{color:#6B1717 !important}
.d-info {background:#EBF5FB;border-left:4px solid #2471A3}
.d-info strong{color:#154360 !important}
.d-info span{color:#1A4F72 !important}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏛️ Domognostics Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">IAQ Analysis · Decision Tree · Student Satisfaction</p>', unsafe_allow_html=True)
st.divider()

# SIDEBAR 
with st.sidebar:
    st.header("⚙️ Ρυθμίσεις")

    # 1. FOLDER
    st.subheader("1. 📁 Φάκελος Δεδομένων")
    st.caption("Επικόλλησε την πλήρη διαδρομή του φακέλου που περιέχει drakos/, tasos/, Tofis/")

    folder_input = st.text_input(
        "Διαδρομή φακέλου",
        value=st.session_state.get('folder_path',''),
        placeholder=r"C:\Users\user\OneDrive\Desktop\data",
        label_visibility="collapsed",
    ).strip().strip('"').strip("'").rstrip('/\\')

    folder_ok = False
    if folder_input:
        if os.path.isdir(folder_input):
            expected=['drakos','tasos','Tofis','tofis']
            found_subs=[s for s in expected if os.path.isdir(os.path.join(folder_input,s))]
            if found_subs:
                st.success(f"✅ Βρέθηκε: {', '.join(found_subs)}")
                folder_ok = True
                st.session_state['folder_path'] = folder_input
                sensors, feedbacks = scan_folder_inventory(folder_input)
                with st.expander(f"📂 {len(sensors)} sensors · {len(feedbacks)} feedbacks"):
                    for p in sensors:
                        st.markdown(f"🔬 `{os.path.relpath(p,folder_input)}`")
                    for p in feedbacks:
                        st.markdown(f"📋 `{os.path.relpath(p,folder_input)}`")
            else:
                st.warning("⚠️ Δεν βρέθηκαν υποφάκελοι drakos/tasos/Tofis")
        else:
            st.error("❌ Η διαδρομή δεν υπάρχει")

    st.divider()

    # 2. ROOM / PART
    st.subheader("2. 🏛️ Αίθουσα")
    room = st.selectbox("Αίθουσα", list(ROOM_STRUCTURE.keys()))
    part = st.selectbox("Τμήμα / Θέση", ROOM_STRUCTURE[room])

    # Quick match preview
    if folder_ok:
        mn = st.session_state.get('_prev_month','10')
        sp = find_sensor_path(folder_input, room, part, mn)
        fp = find_feedback_path(folder_input, room, part, mn)
        st.caption(f"{'✅' if sp else '❌'} Sensor: `{os.path.basename(sp) if sp else 'ΔΕΝ ΒΡΕΘΗΚΕ'}`")
        st.caption(f"{'✅' if fp else '❌'} Feedback: `{os.path.basename(fp) if fp else 'ΔΕΝ ΒΡΕΘΗΚΕ'}`")

    st.divider()

    # 3. DATE / TIME
    st.subheader("3. 🗓️ Ημερομηνία & Ώρα")
    sel_date = st.date_input("Ημερομηνία", value=date(2025,10,1))
    st.session_state['_prev_month'] = sel_date.strftime('%m')
    col_h,col_m = st.columns(2)
    with col_h:
        hour = st.selectbox(
            "Ώρα",
            options=list(range(8, 23)),
            index=2,
            format_func=lambda x: f"{x:02d}",
        )
    with col_m:
        minute = st.selectbox(
            "Λεπτά",
            options=list(range(0, 60)),
            index=0,
            format_func=lambda x: f"{x:02d}",
        )

    # Updated preview with correct month
    if folder_ok:
        mn2 = sel_date.strftime('%m')
        sp2 = find_sensor_path(folder_input, room, part, mn2)
        fp2 = find_feedback_path(folder_input, room, part, mn2)
        st.markdown("**Αντιστοίχιση για επιλεγμένη ημ/νία:**")
        st.caption(f"{'✅' if sp2 else '❌'} `{os.path.basename(sp2) if sp2 else 'ΔΕΝ ΒΡΕΘΗΚΕ'}`")
        st.caption(f"{'✅' if fp2 else '❌'} `{os.path.basename(fp2) if fp2 else 'ΔΕΝ ΒΡΕΘΗΚΕ'}`")

    st.divider()
    run_btn = st.button("▶️  RUN ANALYSIS", type="primary",
                        use_container_width=True, disabled=not folder_ok)

# LANDING PAGE 
if not folder_ok:
    st.markdown("## 👈 Βάλε τη διαδρομή του φακέλου δεδομένων στο sidebar")
    st.info("""
**💡 Πώς να βρεις τη διαδρομή (Windows):**

1. Άνοιξε τον Explorer στον φάκελο που περιέχει τα `drakos/`, `tasos/`, `Tofis/`
2. Κλικ στη **γραμμή διαδρομής** 
3. **Ctrl+C** για αντιγραφή
4. **Ctrl+V** μέσα στο πεδίο στο sidebar

**Παράδειγμα:**
```
C:\\Users\\user\\OneDrive\\Desktop\\data
```
""")
    st.stop()

# RUN ANALYSIS
if run_btn:
    with st.spinner("⏳ Ανάλυση δεδομένων..."):
        r = run_analysis(folder_input, room, part, sel_date, hour, minute)
        st.session_state['last_result'] = r

r = st.session_state.get('last_result')
if not r:
    st.info("Πάτα **▶️ RUN ANALYSIS** για να ξεκινήσει η ανάλυση.")
    st.stop()

# ERRORS 
err = r.get('error')
if err == 'missing_files':
    mn = MONTH_FEEDBACK_NAME.get(sel_date.strftime('%m'),'')
    st.error(f"""
❌ **Δεν βρέθηκαν αρχεία** για `{room} / {part}` — Μήνας: `{mn}`

| | Αποτέλεσμα |
|-|-----------|
| Sensor CSV | {"✅ `" + os.path.basename(r['s_path']) + "`" if r['s_path'] else "❌ ΔΕΝ ΒΡΕΘΗΚΕ"} |
| Feedback CSV | {"✅ `" + os.path.basename(r['f_path']) + "`" if r['f_path'] else "❌ ΔΕΝ ΒΡΕΘΗΚΕ"} |

Έλεγξε ότι τα αρχεία υπάρχουν στον σωστό υποφάκελο.
""")
    st.stop()
elif err == 'no_sensor_data':
    st.error(f"❌ Δεν βρέθηκαν sensor δεδομένα για {r['s_start'].strftime('%H:%M')}–{r['s_end'].strftime('%H:%M')}. Δοκίμασε διαφορετική ώρα.")
    st.stop()
elif err:
    st.error(f"❌ Σφάλμα: {err}")
    st.stop()

# STATUS BANNER 
decisions = r['decisions_main'] + r['decisions_common']
has_alert = any(s=='alert' for s,*_ in decisions)
has_warn  = any(s=='warn'  for s,*_ in decisions)

c1,c2,c3 = st.columns(3)
c1.metric("🗓️ Ανάλυση", r['target_time'].strftime('%d/%m/%Y %H:%M'))
c2.metric("🏛️ Αίθουσα", f"{r['room']} / {r['part']}")
c3.metric("🌡️ Κλιματική Περίοδος", "🌤️ Π1 — Θερμή" if r['P1'] else "❄️ Π2 — Ψυχρή")

if has_alert:  st.error("🚨 **ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΑΠΑΙΤΟΥΝΤΑΙ ΑΜΕΣΕΣ ΕΝΕΡΓΕΙΕΣ**")
elif has_warn: st.warning("⚠️ **ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΧΡΕΙΑΖΕΤΑΙ ΠΑΡΑΚΟΛΟΥΘΗΣΗ**")
else:          st.success("✅ **ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΒΕΛΤΙΣΤΕΣ ΣΥΝΘΗΚΕΣ**")

st.divider()

# TABS 
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
    "📊 Μετρήσεις",
    "💡 Suggestions",
    "📈 Γραφήματα",
    "🔄 Σύγκριση Τμημάτων",
    "📅 Χρονική Ανάλυση",
    "📋 Μηνιαία Στατιστική",
    "🏫 Σύγκριση Αιθουσών",
    "📄 PDF",
])

with tab1:
    v = r['v']
    st.subheader(f"Sensor Data  ±90 λεπτά  |  {r['s_start'].strftime('%H:%M')} – {r['s_end'].strftime('%H:%M')}")
    st.caption(f"📁 {r['s_path']}")
    cols = st.columns(4)
    for i,(lbl,key,unit) in enumerate([
        ("🌡️ Θερμοκρασία","T","°C"),("💧 Υγρασία","H","%"),
        ("💨 CO2","C","ppm"),("🧪 VOC","VOC","ppb"),
        ("🌫️ PM1","PM1","μg/m³"),("🌫️ PM2.5","PM25","μg/m³"),
        ("🔊 Θόρυβος","N","dBA"),("🧭 Πίεση","P","hPa"),
    ]):
        val=v.get(key)
        with cols[i%4]:
            st.metric(lbl, f"{val} {unit}" if val not in (None,"N/A") else "N/A")

    st.divider()
    st.subheader(f"Student Satisfaction  ±90 λεπτά  |  {r['f_start'].strftime('%H:%M')} – {r['f_end'].strftime('%H:%M')}")
    st.caption(f"📁 {r['f_path']}")
    tf=r['total_f']; cp=r['comfort_pct']
    if tf==0:
        st.info("ℹ️ Δεν υπάρχουν feedbacks στο χρονικό παράθυρο.")
    elif not r['fb_col']:
        st.warning(f"⚠️ {tf} feedbacks αλλά δεν βρέθηκε στήλη feedback.")
    else:
        f1,f2=st.columns(2)
        f1.metric("📋 Feedbacks", tf)
        f2.metric("😊 Satisfaction", f"{cp:.1f}%")
        st.progress(int(cp))
        df_fb=pd.DataFrame(list(r['category_counts'].items()),columns=['Κατηγορία','Πλήθος'])
        df_fb['%']=(df_fb['Πλήθος']/tf*100).round(1)
        st.dataframe(df_fb.sort_values('Πλήθος',ascending=False).reset_index(drop=True),
                     use_container_width=True, hide_index=True)

with tab2:
    P1=r['P1']
    st.subheader("💡 Suggestions — " + ("🌤️ Περίοδος 1: Θερμή [Σεπ – 20 Νοε]" if P1 else "❄️ Περίοδος 2: Ψυχρή [20 Νοε – Ιαν]"))
    CSS={'ok':'d-ok','warn':'d-warn','alert':'d-alert','info':'d-info'}
    ICONS={'ok':'✅','warn':'⚠️','alert':'🚨','info':'ℹ️'}
    for status,param,msg in decisions:
        st.markdown(
            f'<div class="decision-row {CSS.get(status,"")}">'
            f'<strong>{ICONS.get(status,"")} {param}</strong><br>'
            f'<span style="color:#555">→ {msg}</span></div>',
            unsafe_allow_html=True)

    if r['total_f']>=1 and r['comfort_pct'] is not None and r['fb_col']:
        st.divider(); st.subheader("🔗 Συσχέτιση Μετρήσεων – Αντίληψης Φοιτητών")
        NEGATIVE={'too hot','too cold','too dry','too humid','irritating','unpleasant'}
        cc=r['category_counts']; tf=r['total_f']
        neg_l=[l for l in cc if l.lower() in NEGATIVE]; neg_n=sum(cc[l] for l in neg_l)
        s_ok=not any(s in ('alert','warn') for s,*_ in r['decisions_main']); cp=r['comfort_pct']
        if s_ok and neg_n>0:
            st.warning(f"⚠️ **ΑΣΥΜΦΩΝΙΑ**: Αποδεκτές μετρήσεις αλλά {neg_n}/{tf} φοιτητές ({neg_n/tf*100:.0f}%) δηλώνουν δυσφορία.\n\n→ Πιθανή αιτία: τοπική ανομοιογένεια (θέση καθίσματος, εγγύτητα σε A/C).")
        elif not s_ok and cp==100 and tf>=2:
            st.info(f"ℹ️ **ΠΑΡΑΤΗΡΗΣΗ**: Εκτός ορίων μετρήσεις αλλά {tf} φοιτητές δηλώνουν 100% άνεση.")
        elif not s_ok and neg_n>0:
            st.error(f"🚨 **ΣΥΜΦΩΝΙΑ ΠΡΟΒΛΗΜΑΤΟΣ**: Εκτός ορίων ΚΑΙ {neg_n}/{tf} φοιτητές ({neg_n/tf*100:.0f}%) δηλώνουν δυσφορία → Άμεση παρέμβαση.")
        else:
            st.success(f"✅ **ΣΥΜΦΩΝΙΑ**: Αποδεκτές μετρήσεις & {cp:.0f}% ικανοποίηση.")

with tab3:
    v=r['v']
    cc1,cc2=st.columns([3,2])
    with cc1:
        st.markdown("**Sensor Data & Satisfaction Index**")
        st.image(chart_dual_axis(v,r['comfort_pct'],r['target_time'],r['season_id']))
    with cc2:
        if r['total_f']>0 and r['category_counts']:
            st.markdown("**Κατανομή Feedback**")
            pie=chart_pie(r['category_counts'],r['total_f'],r['comfort_pct'] or 0)
            if pie: st.image(pie)
    st.markdown("**Correlation Heatmap**")
    st.image(chart_heatmap(v,r['category_counts'],r['total_f']))

    st.divider()

    # Heatmap Ημέρας / Ώρας 
    st.markdown("### 🗓️ Heatmap Ημέρας / Ώρας")
    month_label_hm = f"{MONTH_FEEDBACK_NAME.get(sel_date.strftime('%m'),'')} {sel_date.year}"
    st.caption(f"Άξονας X = Ημέρα Μήνα · Άξονας Y = Ώρα · Χρώμα = Τιμή παραμέτρου  "
               f"(🟢 εντός ορίων · 🔴 εκτός ορίων)")

    if not r['s_path']:
        st.warning("⚠️ Δεν βρέθηκε sensor CSV για heatmap.")
    else:
        LIMITS_HM_P1 = {'T':(23,27),'H':(40,60),'C':(400,1000),'N':(0,35),'PM25':(0,12)}
        LIMITS_HM_P2 = {'T':(20,24),'H':(30,50),'C':(400,1200),'N':(0,35),'PM25':(0,12)}
        LIMITS_HM = LIMITS_HM_P1 if r['P1'] else LIMITS_HM_P2

        HM_PARAMS = [
            ('T',    ['Temperature'],            'Θερμοκρασία', '°C'),
            ('H',    ['Humidity'],               'Υγρασία',     '%'),
            ('C',    ['Carbon Dioxide','CO2'],   'CO2',         'ppm'),
            ('N',    ['Noise'],                  'Θόρυβος',     'dBA'),
            ('PM25', ['PM2.5','PM 2.5','pm2.5'], 'PM2.5',       'μg/m³'),
        ]

        hm_param_choice = st.selectbox(
            "Επέλεξε παράμετρο:",
            options=[p[2] for p in HM_PARAMS],
            index=0,
            key="hm_param_tab3"
        )
        selected_hm = next(p for p in HM_PARAMS if p[2]==hm_param_choice)
        key_hm, names_hm, label_hm, unit_hm = selected_hm
        limits_hm = LIMITS_HM.get(key_hm)

        with st.spinner(f"⏳ Δημιουργία heatmap {label_hm}..."):
            hm_buf, hm_err = chart_day_hour_heatmap(
                r['s_path'], r['f_path'],
                key_hm, names_hm, label_hm, unit_hm,
                limits_hm, month_label_hm
            )
        if hm_err:
            st.warning(f"⚠️ {hm_err}")
        else:
            st.image(hm_buf, use_container_width=True)

        if r['f_path']:
            st.markdown("**😊 Heatmap Student Satisfaction**")
            with st.spinner("Δημιουργία satisfaction heatmap..."):
                sat_hm_buf = chart_satisfaction_heatmap(r['f_path'], month_label_hm)
            if sat_hm_buf:
                st.image(sat_hm_buf, use_container_width=True)
            else:
                st.info("ℹ️ Δεν υπάρχουν αρκετά feedbacks για heatmap.")

with tab4:
    # Μόνο για DRAKOS και TASOS
    if r['room'] == "TOFIS":
        st.info("ℹ️ Η αίθουσα TOFIS δεν έχει χωρική διάταξη — η σύγκριση δεν εφαρμόζεται.")
    else:
        parts_list = ROOM_STRUCTURE[r['room']]
        target_time = r['target_time']
        P1 = r['P1']

        st.subheader(f"🔄 Σύγκριση Τμημάτων — {r['room']}")
        st.caption(f"📅 {target_time.strftime('%d/%m/%Y %H:%M')}  |  "
                   f"{'🌤️ Περίοδος 1 — Θερμή' if P1 else '❄️ Περίοδος 2 — Ψυχρή'}")

        with st.spinner("⏳ Φόρτωση δεδομένων όλων των τμημάτων..."):
            all_parts = [
                load_part_data(folder_input, r['room'], p, sel_date, hour, minute)
                for p in parts_list
            ]

        # Status table 
        st.markdown("#### 📋 Επισκόπηση Αρχείων")
        rows = []
        for d in all_parts:
            rows.append({
                'Τμήμα':      d['part'].replace('_',' '),
                'Sensor CSV': '✅ ' + os.path.basename(d['s_path']) if d['s_path'] else '❌ ΔΕΝ ΒΡΕΘΗΚΕ',
                'Feedback CSV': '✅ ' + os.path.basename(d['f_path']) if d['f_path'] else '❌ ΔΕΝ ΒΡΕΘΗΚΕ',
                'Δεδομένα': '✅' if d['found_sensor'] else '⚠️ Εκτός ωραρίου',
                'Feedbacks': str(d['total_f']),
                'Satisfaction': f"{d['comfort_pct']:.1f}%" if d['comfort_pct'] is not None else 'N/A',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # Metrics comparison table 
        st.markdown("#### 📊 Συγκριτικός Πίνακας Μετρήσεων")

        #  χρωματισμός ορίων ανά περίοδο
        LIMITS_P1 = {'T':(23,27),'H':(40,60),'C':(0,1000),'VOC':(0,100),'PM1':(0,10),'PM25':(0,12),'N':(0,35),'P':(980,1050)}
        LIMITS_P2 = {'T':(20,24),'H':(30,50),'C':(0,1200),'VOC':(0,100),'PM1':(0,10),'PM25':(0,12),'N':(0,35),'P':(980,1050)}
        LIMITS = LIMITS_P1 if P1 else LIMITS_P2

        PARAMS = [
            ('🌡️ Θερμοκρασία','T','°C'),
            ('💧 Υγρασία','H','%'),
            ('💨 CO2','C','ppm'),
            ('🧪 VOC','VOC','ppb'),
            ('🌫️ PM1','PM1','μg/m³'),
            ('🌫️ PM2.5','PM25','μg/m³'),
            ('🔊 Θόρυβος','N','dBA'),
            ('🧭 Πίεση','P','hPa'),
        ]

        def status_icon(key, val):
            if val in (None,"N/A"): return "➖"
            lo,hi = LIMITS.get(key,(0,9999))
            return "✅" if lo <= val <= hi else "🚨"

        # Build table
        tbl_data = {'Παράμετρος': [f"{lbl} ({unit})" for lbl,_,unit in PARAMS]}
        for d in all_parts:
            col_name = d['part'].replace('_',' ')
            tbl_data[col_name] = []
            for _,key,unit in PARAMS:
                val = d['v'].get(key)
                icon = status_icon(key, val)
                display = f"{icon} {val}" if val not in (None,"N/A") else "➖ N/A"
                tbl_data[col_name].append(display)

        # Satisfaction row
        tbl_data['Παράμετρος'].append('😊 Satisfaction')
        for d in all_parts:
            col_name = d['part'].replace('_',' ')
            cp = d['comfort_pct']
            if cp is None:
                tbl_data[col_name].append("➖ N/A")
            elif cp >= 65:
                tbl_data[col_name].append(f"✅ {cp:.1f}%")
            elif cp >= 40:
                tbl_data[col_name].append(f"⚠️ {cp:.1f}%")
            else:
                tbl_data[col_name].append(f"🚨 {cp:.1f}%")

        st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True)

        st.divider()

        # Charts 
        st.markdown("#### 📈 Γραφήματα Σύγκρισης")

        # Radar chart 
        st.markdown("**🕸️ IAQ Score Radar — Συνολική Εικόνα**")
        st.caption("Κανονικοποιημένος δείκτης ποιότητας (1.0 = Βέλτιστο για την περίοδο)")
        radar_buf = chart_radar_comparison(all_parts, P1)
        col_r1, col_r2, col_r3 = st.columns([1,2,1])
        with col_r2:
            st.image(radar_buf, use_container_width=True)

        st.divider()

        # Parameter charts σε grid
        st.markdown("**📊 Σύγκριση ανά Παράμετρο**")

        CHART_PARAMS = [
            ('T',  'Θερμοκρασία', '°C',    LIMITS['T']),
            ('H',  'Υγρασία',     '%',     LIMITS['H']),
            ('C',  'CO2',         'ppm',   LIMITS['C']),
            ('VOC','VOC',         'ppb',   (0,100)),
            ('PM25','PM2.5',      'μg/m³', (0,12)),
            ('N',  'Θόρυβος',    'dBA',   (0,35)),
        ]

        # 2 charts 
        for i in range(0, len(CHART_PARAMS), 2):
            col_a, col_b = st.columns(2)
            for col, idx in [(col_a, i), (col_b, i+1)]:
                if idx < len(CHART_PARAMS):
                    key, label, unit, good_range = CHART_PARAMS[idx]
                    with col:
                        buf = chart_comparison_bars(all_parts, key, label, unit, good_range)
                        st.image(buf, use_container_width=True)

        st.divider()

        # Satisfaction chart
        st.markdown("**😊 Student Satisfaction Index — Σύγκριση**")
        sat_buf = chart_satisfaction_comparison(all_parts)
        col_s1, col_s2, col_s3 = st.columns([1,2,1])
        with col_s2:
            st.image(sat_buf, use_container_width=True)

        # Αυτόματα συμπεράσματα 
        st.divider()
        st.markdown("#### 🏆 Αυτόματα Συμπεράσματα")

        # Βρίσκει το καλύτερο τμήμα βάσει IAQ score
        def iaq_score(d):
            score = 0; count = 0
            for key in ['T','H','C','VOC','PM25','N']:
                val = d['v'].get(key)
                if val in (None,"N/A"): continue
                lo,hi = LIMITS.get(key,(0,9999))
                score += 1 if lo<=val<=hi else 0
                count += 1
            return score/count if count>0 else 0

        scored = [(d, iaq_score(d)) for d in all_parts if d['found_sensor']]
        if scored:
            best_iaq  = max(scored, key=lambda x: x[1])
            worst_iaq = min(scored, key=lambda x: x[1])

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.success(f"🏆 **Καλύτερη IAQ:** {best_iaq[0]['part'].replace('_',' ')} "
                           f"({best_iaq[1]*100:.0f}% παράμετροι εντός ορίων)")
            with col_c2:
                if worst_iaq[0]['part'] != best_iaq[0]['part']:
                    st.error(f"⚠️ **Χειρότερη IAQ:** {worst_iaq[0]['part'].replace('_',' ')} "
                             f"({worst_iaq[1]*100:.0f}% παράμετροι εντός ορίων)")

        # Καλύτερο satisfaction
        sat_data = [(d, d['comfort_pct']) for d in all_parts if d['comfort_pct'] is not None]
        if sat_data:
            best_sat  = max(sat_data, key=lambda x: x[1])
            worst_sat = min(sat_data, key=lambda x: x[1])
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.success(f"😊 **Υψηλότερη Satisfaction:** {best_sat[0]['part'].replace('_',' ')} "
                           f"({best_sat[1]:.1f}%)")
            with col_d2:
                if worst_sat[0]['part'] != best_sat[0]['part']:
                    pct = worst_sat[1]
                    icon = "⚠️" if pct >= 40 else "🚨"
                    st.warning(f"{icon} **Χαμηλότερη Satisfaction:** {worst_sat[0]['part'].replace('_',' ')} "
                               f"({pct:.1f}%)")

        # Εντοπισμός διαφορών θερμοκρασίας μεταξύ τμημάτων
        temp_vals = [(d['part'], d['v'].get('T')) for d in all_parts
                     if d['v'].get('T') not in (None,"N/A")]
        if len(temp_vals) >= 2:
            max_t = max(temp_vals, key=lambda x: x[1])
            min_t = min(temp_vals, key=lambda x: x[1])
            diff  = round(max_t[1] - min_t[1], 1)
            if diff >= 1.0:
                st.warning(
                    f"🌡️ **Ανομοιογένεια θερμοκρασίας:** Διαφορά **{diff} °C** μεταξύ "
                    f"{max_t[0].replace('_',' ')} ({max_t[1]} °C) και "
                    f"{min_t[0].replace('_',' ')} ({min_t[1]} °C). "
                    f"→ Πιθανά προβλήματα κατανομής θέρμανσης/ψύξης."
                )


#  TIME SERIES  – για ολόκληρη τηην μέρα

def chart_time_series(s_path, f_path, sel_date, params_to_plot, P1):
    """
    Γράφημα χρονοσειράς για μια ολόκληρη μέρα.
    params_to_plot: list of (col_name, label, color, unit)
    """
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    df_s = read_csv_path(s_path)
    if df_s is None:
        return None, "Αδυναμία ανάγνωσης sensor CSV."

    df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
    day_start = pd.Timestamp(sel_date.year, sel_date.month, sel_date.day, 0, 0)
    day_end   = pd.Timestamp(sel_date.year, sel_date.month, sel_date.day, 23, 59)
    df_day = df_s[(df_s['Timestamp'] >= day_start) & (df_s['Timestamp'] <= day_end)].copy()

    if df_day.empty:
        return None, "Δεν βρέθηκαν δεδομένα για αυτή την ημέρα."

    m_col = next((c for c in df_day.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                 next((c for c in df_day.columns if 'meas' in c.lower()), None))
    v_col = next((c for c in df_day.columns if c.lower() == 'value'),
                 next((c for c in df_day.columns if 'valu' in c.lower()), None))

    if not m_col or not v_col:
        return None, f"Δεν βρέθηκαν στήλες measurement/value."

    n_params = len(params_to_plot)
    fig, axes = plt.subplots(n_params, 1, figsize=(11, 3.2 * n_params), sharex=True)
    fig.patch.set_facecolor('#F8F9FA')
    if n_params == 1:
        axes = [axes]

    # Ορισμός ορίων ανά παράμετρο
    LIMITS_P1 = {'Temperature':(23,27),'Humidity':(40,60),'Carbon Dioxide':(0,1000),
                 'VOC':(0,100),'PM2.5':(0,12),'Noise':(0,35)}
    LIMITS_P2 = {'Temperature':(20,24),'Humidity':(30,50),'Carbon Dioxide':(0,1200),
                 'VOC':(0,100),'PM2.5':(0,12),'Noise':(0,35)}
    LIMITS = LIMITS_P1 if P1 else LIMITS_P2

    for ax, (meas_name, label, color, unit) in zip(axes, params_to_plot):
        ax.set_facecolor('#FFFFFF')
        mask = df_day[m_col].astype(str).str.strip().str.lower() == meas_name.lower()
        df_p = df_day[mask].copy()
        df_p[v_col] = pd.to_numeric(df_p[v_col], errors='coerce')
        df_p = df_p.dropna(subset=[v_col]).sort_values('Timestamp')

        if df_p.empty:
            ax.text(0.5, 0.5, f'{label}: Δεν υπάρχουν δεδομένα',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='#7F8C8D', style='italic')
        else:
            # Resample ανά 5 λεπτά πιο smooth γράφημα
            df_p = df_p.set_index('Timestamp')
            df_resampled = df_p[v_col].resample('5min').mean().interpolate()
            times = df_resampled.index
            vals  = df_resampled.values

            ax.plot(times, vals, color=color, linewidth=1.8, alpha=0.9)
            ax.fill_between(times, vals, alpha=0.12, color=color)

            # Ζώνη ορίων
            if meas_name in LIMITS:
                lo, hi = LIMITS[meas_name]
                ax.axhspan(lo, hi, alpha=0.08, color='#27AE60', label=f'Βέλτιστο ({lo}–{hi} {unit})')
                ax.axhline(y=lo, color='#27AE60', linestyle=':', linewidth=1, alpha=0.6)
                ax.axhline(y=hi, color='#E74C3C', linestyle=':', linewidth=1, alpha=0.6)

            # min / Max / Mean 
            mean_v = df_resampled.mean()
            max_v  = df_resampled.max()
            min_v  = df_resampled.min()
            ax.annotate(f'Max: {max_v:.1f}', xy=(times[vals.argmax()], max_v),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, color='#C0392B', fontweight='bold')
            ax.annotate(f'Min: {min_v:.1f}', xy=(times[vals.argmin()], min_v),
                        xytext=(5, -12), textcoords='offset points',
                        fontsize=7, color='#2980B9', fontweight='bold')
            ax.axhline(y=mean_v, color=color, linestyle='--', linewidth=1,
                       alpha=0.5, label=f'Μέσος: {mean_v:.1f} {unit}')
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

        ax.set_ylabel(f'{label}\n({unit})', fontsize=8.5, color='#2C3E50')
        ax.tick_params(axis='both', labelsize=7.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4, color='#BDC3C7')

    axes[-1].set_xlabel('Ώρα', fontsize=9, color='#2C3E50')
    import matplotlib.dates as mdates
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45, ha='right', fontsize=7.5)

    period_str = 'Περίοδος 1 — Θερμή' if P1 else 'Περίοδος 2 — Ψυχρή'
    fig.suptitle(f"Χρονική Εξέλιξη Παραμέτρων  |  {sel_date.strftime('%d/%m/%Y')}  |  {period_str}",
                 fontsize=11, fontweight='bold', color='#2C3E50', y=1.01)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0)
    return buf, None


def chart_satisfaction_timeline(f_path, sel_date):
    """Scatter plot feedback satisfaction κατά τη διάρκεια της μέρας."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    df_f = read_csv_path(f_path)
    if df_f is None:
        return None

    df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')
    day_start = pd.Timestamp(sel_date.year, sel_date.month, sel_date.day, 0, 0)
    day_end   = pd.Timestamp(sel_date.year, sel_date.month, sel_date.day, 23, 59)
    df_day = df_f[(df_f['Timestamp'] >= day_start) & (df_f['Timestamp'] <= day_end)].copy()

    if df_day.empty:
        return None

    fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in df_day.columns else None
    if not fb_col:
        fb_col = next((c for c in df_day.columns
                       if any(k in c.lower() for k in ('temp','feed','comfort'))), None)
    if not fb_col:
        return None

    df_day['weight'] = df_day[fb_col].astype(str).str.strip().str.lower().map(
        lambda x: FEEDBACK_WEIGHTS.get(x, 0.5)
    )

    color_map = {1.0:'#27AE60', 0.5:'#F39C12', 0.0:'#E74C3C'}
    label_map = {1.0:'Θετικό (Comfortable/Pleasant)',
                 0.5:'Ουδέτερο (Neutral/Noticeable)',
                 0.0:'Αρνητικό (Too Hot/Cold/…)'}

    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#FFFFFF')

    for weight, grp in df_day.groupby('weight'):
        ax.scatter(grp['Timestamp'], [weight]*len(grp),
                   c=color_map.get(weight,'#95A5A6'), s=80, alpha=0.8,
                   label=label_map.get(weight, str(weight)), zorder=3)

    # Rolling satisfaction
    df_day_sorted = df_day.sort_values('Timestamp')
    if len(df_day_sorted) >= 3:
        roll = df_day_sorted.set_index('Timestamp')['weight'].rolling('2h', center=True).mean()
        ax2 = ax.twinx()
        ax2.plot(roll.index, roll.values*100, color='#2980B9', linewidth=2,
                 linestyle='--', alpha=0.7, label='Rolling Satisfaction (2h)')
        ax2.set_ylabel('Satisfaction %', fontsize=8, color='#2980B9')
        ax2.set_ylim(-10, 115)
        ax2.tick_params(axis='y', labelcolor='#2980B9', labelsize=7.5)
        ax2.spines['top'].set_visible(False)
        ax2.legend(loc='upper left', fontsize=7.5, framealpha=0.8)

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['Αρνητικό', 'Ουδέτερο', 'Θετικό'], fontsize=8)
    ax.set_ylabel('Κατηγορία Feedback', fontsize=8.5, color='#2C3E50')
    ax.set_xlabel('Ώρα', fontsize=9)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45, ha='right', fontsize=7.5)
    ax.set_title(f"Εξέλιξη Feedback Φοιτητών  |  {sel_date.strftime('%d/%m/%Y')}",
                 fontsize=10, pad=8, color='#2C3E50', fontweight='bold')
    plt.tight_layout(pad=1.2)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0)
    return buf



#  MONTHLY STATS
def compute_monthly_stats(s_path, f_path):
    """Υπολογίζει μηνιαία στατιστικά για sensor και feedbacks."""
    stats = {}

    # Sensor stats
    df_s = read_csv_path(s_path)
    if df_s is not None:
        df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
        m_col = next((c for c in df_s.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                     next((c for c in df_s.columns if 'meas' in c.lower()), None))
        v_col = next((c for c in df_s.columns if c.lower() == 'value'),
                     next((c for c in df_s.columns if 'valu' in c.lower()), None))

        if m_col and v_col:
            SENSOR_MAP = {
                'Θερμοκρασία':   ['Temperature'],
                'Υγρασία':       ['Humidity'],
                'CO2':           ['Carbon Dioxide','CO2'],
                'VOC':           ['VOC','Volatile Organic Compounds'],
                'PM1':           ['PM1','PM 1','PM1.0'],
                'PM2.5':         ['PM2.5','PM 2.5','pm2.5'],
                'Θόρυβος':       ['Noise'],
                'Πίεση':         ['Pressure'],
            }
            UNITS = {'Θερμοκρασία':'°C','Υγρασία':'%','CO2':'ppm','VOC':'ppb',
                     'PM1':'μg/m³','PM2.5':'μg/m³','Θόρυβος':'dBA','Πίεση':'hPa'}

            sensor_rows = []
            for param, names in SENSOR_MAP.items():
                mask = df_s[m_col].astype(str).str.strip().str.lower().isin(
                    [n.lower() for n in names])
                vals = pd.to_numeric(df_s.loc[mask, v_col], errors='coerce').dropna()
                if not vals.empty:
                    sensor_rows.append({
                        'Παράμετρος': f"{param} ({UNITS.get(param,'')})",
                        'Μέσος Όρος': round(vals.mean(), 2),
                        'Τυπ. Απόκλιση': round(vals.std(), 2),
                        'Ελάχιστο': round(vals.min(), 2),
                        'Μέγιστο': round(vals.max(), 2),
                        'Μεσαία Τιμή': round(vals.median(), 2),
                        'N Μετρήσεις': len(vals),
                    })
                else:
                    sensor_rows.append({
                        'Παράμετρος': f"{param} ({UNITS.get(param,'')})",
                        'Μέσος Όρος':'N/A','Τυπ. Απόκλιση':'N/A',
                        'Ελάχιστο':'N/A','Μέγιστο':'N/A',
                        'Μεσαία Τιμή':'N/A','N Μετρήσεις': 0,
                    })
            stats['sensor_df'] = pd.DataFrame(sensor_rows)
            stats['total_sensor_records'] = len(df_s)
            stats['date_range'] = (df_s['Timestamp'].min(), df_s['Timestamp'].max())

    # Feedback stats
    df_f = read_csv_path(f_path)
    if df_f is not None:
        df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')
        fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in df_f.columns else None
        if not fb_col:
            fb_col = next((c for c in df_f.columns
                           if any(k in c.lower() for k in ('temp','feed','comfort'))), None)
        if fb_col:
            total = len(df_f)
            cats  = df_f[fb_col].astype(str).str.strip().value_counts()
            ws    = sum(FEEDBACK_WEIGHTS.get(l.lower(),0.0)*cnt for l,cnt in cats.items())
            monthly_sat = round((ws/total)*100, 1) if total > 0 else None

            fb_rows = []
            for label, count in cats.items():
                fb_rows.append({
                    'Κατηγορία': label,
                    'Πλήθος': count,
                    '%': round(count/total*100, 1),
                    'Βάρος': FEEDBACK_WEIGHTS.get(label.lower(), 0.5),
                })
            stats['feedback_df'] = pd.DataFrame(fb_rows)
            stats['total_feedbacks'] = total
            stats['monthly_satisfaction'] = monthly_sat
            stats['fb_col'] = fb_col

    return stats


def chart_monthly_boxplot(s_path, P1):
    """Box plot κατανομής τιμών ανά παράμετρο για τον μήνα."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    df_s = read_csv_path(s_path)
    if df_s is None: return None

    df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
    m_col = next((c for c in df_s.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                 next((c for c in df_s.columns if 'meas' in c.lower()), None))
    v_col = next((c for c in df_s.columns if c.lower() == 'value'),
                 next((c for c in df_s.columns if 'valu' in c.lower()), None))
    if not m_col or not v_col: return None

    PARAMS = [
        ('Temperature','Θερμοκρασία','°C','#E74C3C',(23,27) if P1 else (20,24)),
        ('Humidity','Υγρασία','%','#3498DB',(40,60) if P1 else (30,50)),
        ('Carbon Dioxide','CO2','ppm','#27AE60',(400,1000) if P1 else (400,1200)),
        ('Noise','Θόρυβος','dBA','#F39C12',(0,35)),
    ]

    data_dict = {}
    for meas, label, unit, color, limits in PARAMS:
        mask = df_s[m_col].astype(str).str.strip().str.lower() == meas.lower()
        vals = pd.to_numeric(df_s.loc[mask, v_col], errors='coerce').dropna()
        if not vals.empty:
            data_dict[(label, unit, color, limits)] = vals.values

    if not data_dict:
        return None

    fig, axes = plt.subplots(1, len(data_dict), figsize=(4*len(data_dict), 4.5))
    fig.patch.set_facecolor('#F8F9FA')
    if len(data_dict) == 1: axes = [axes]

    for ax, ((label, unit, color, limits), vals) in zip(axes, data_dict.items()):
        ax.set_facecolor('#FFFFFF')
        bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                        medianprops=dict(color='white', linewidth=2.5),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4,
                                        markerfacecolor=color))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.75)

        # Ζώνη βέλτιστων ορίων
        lo, hi = limits
        ax.axhspan(lo, hi, alpha=0.1, color='#27AE60')
        ax.axhline(y=lo, color='#27AE60', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=hi, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.7)

        # Stats text
        ax.text(1.32, np.median(vals), f'Μεσ: {np.median(vals):.1f}',
                va='center', fontsize=7.5, color='#2C3E50', fontweight='bold',
                transform=ax.get_yaxis_transform())

        ax.set_title(f'{label}\n({unit})', fontsize=9, color='#2C3E50', fontweight='bold')
        ax.set_xticks([]); ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # ποσοστό εντός ορίων
        in_range = ((vals >= lo) & (vals <= hi)).mean() * 100
        ax.set_xlabel(f'{in_range:.0f}% εντός ορίων', fontsize=8,
                      color='#27AE60' if in_range>=70 else '#E74C3C')

    fig.suptitle('Κατανομή Τιμών ανά Παράμετρο — Μηνιαία Ανάλυση',
                 fontsize=11, fontweight='bold', color='#2C3E50')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0)
    return buf


with tab5:
    # Χρονική Ανάλυση 
    st.subheader(f"📅 Χρονική Ανάλυση — {sel_date.strftime('%d/%m/%Y')}")
    st.caption(f"🏛️ {r['room']} / {r['part']}  |  "
               f"{'🌤️ Περίοδος 1 — Θερμή' if r['P1'] else '❄️ Περίοδος 2 — Ψυχρή'}")

    if not r['s_path']:
        st.error("❌ Δεν βρέθηκε sensor CSV για αυτή τη ρύθμιση.")
    else:
        # Επιλογή παραμέτρων
        st.markdown("**Επέλεξε παραμέτρους για εμφάνιση:**")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1: show_T   = st.checkbox("🌡️ Θερμοκρασία", value=True)
        with col_p2: show_H   = st.checkbox("💧 Υγρασία", value=True)
        with col_p3: show_CO2 = st.checkbox("💨 CO2", value=True)
        with col_p4: show_N   = st.checkbox("🔊 Θόρυβος", value=False)

        col_p5, col_p6, col_p7, col_p8 = st.columns(4)
        with col_p5: show_VOC  = st.checkbox("🧪 VOC", value=False)
        with col_p6: show_PM25 = st.checkbox("🌫️ PM2.5", value=False)
        with col_p7: show_PM1  = st.checkbox("🌫️ PM1", value=False)
        with col_p8: show_P    = st.checkbox("🧭 Πίεση", value=False)

        PARAM_DEFS = [
            ('Temperature',    'Θερμοκρασία', '#E74C3C', '°C',    show_T),
            ('Humidity',       'Υγρασία',      '#3498DB', '%',     show_H),
            ('Carbon Dioxide', 'CO2',          '#27AE60', 'ppm',   show_CO2),
            ('Noise',          'Θόρυβος',      '#F39C12', 'dBA',   show_N),
            ('VOC',            'VOC',          '#9B59B6', 'ppb',   show_VOC),
            ('PM2.5',          'PM2.5',        '#1ABC9C', 'μg/m³', show_PM25),
            ('PM1',            'PM1',          '#E67E22', 'μg/m³', show_PM1),
            ('Pressure',       'Πίεση',        '#95A5A6', 'hPa',   show_P),
        ]
        selected_params = [(m,l,c,u) for m,l,c,u,sel in PARAM_DEFS if sel]

        if not selected_params:
            st.info("ℹ️ Επέλεξε τουλάχιστον μία παράμετρο.")
        else:
            with st.spinner("⏳ Φόρτωση χρονοσειράς..."):
                ts_buf, ts_err = chart_time_series(
                    r['s_path'], r['f_path'], sel_date, selected_params, r['P1'])

            if ts_err:
                st.warning(f"⚠️ {ts_err}")
            else:
                st.image(ts_buf, use_container_width=True)

        # Feedback timeline
        if r['f_path']:
            st.divider()
            st.markdown("** Εξέλιξη Feedback Φοιτητών κατά τη Διάρκεια της Μέρας**")
            with st.spinner("Φόρτωση feedback timeline..."):
                fb_tl = chart_satisfaction_timeline(r['f_path'], sel_date)
            if fb_tl:
                st.image(fb_tl, use_container_width=True)
            else:
                st.info("ℹ️ Δεν υπάρχουν feedbacks για αυτή την ημέρα.")


with tab6:
    # Μηνιαία Στατιστική 
    st.subheader("📋 Μηνιαία Στατιστική Ανάλυση")
    month_name = MONTH_FEEDBACK_NAME.get(sel_date.strftime('%m'), '')
    st.caption(f"🏛️ {r['room']} / {r['part']}  |  📅 {month_name} {sel_date.year}")

    if not r['s_path'] or not r['f_path']:
        st.error("❌ Δεν βρέθηκαν αρχεία για αυτή τη ρύθμιση.")
    else:
        with st.spinner("⏳ Υπολογισμός μηνιαίων στατιστικών..."):
            mstats = compute_monthly_stats(r['s_path'], r['f_path'])

        # Sensor Stats 
        st.markdown("#### 🔬 Στατιστικά Αισθητήρα")
        if 'date_range' in mstats:
            dr = mstats['date_range']
            col_dr1, col_dr2, col_dr3 = st.columns(3)
            col_dr1.metric("📅 Από", dr[0].strftime('%d/%m/%Y %H:%M') if pd.notna(dr[0]) else 'N/A')
            col_dr2.metric("📅 Έως", dr[1].strftime('%d/%m/%Y %H:%M') if pd.notna(dr[1]) else 'N/A')
            col_dr3.metric("📊 Σύνολο Μετρήσεων", mstats.get('total_sensor_records', 'N/A'))

        if 'sensor_df' in mstats:
            st.dataframe(mstats['sensor_df'], use_container_width=True, hide_index=True)

            # Box plot
            st.markdown("** Box Plot Κατανομής Τιμών**")
            st.caption("Πράσινη ζώνη = βέλτιστο εύρος για την περίοδο")
            with st.spinner("Δημιουργία box plot..."):
                bp_buf = chart_monthly_boxplot(r['s_path'], r['P1'])
            if bp_buf:
                st.image(bp_buf, use_container_width=True)

        st.divider()

        # Feedback Stats 
        st.markdown("####  Στατιστικά Feedbacks")
        if 'total_feedbacks' in mstats:
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("📋 Σύνολο Feedbacks Μήνα", mstats['total_feedbacks'])
            ms = mstats.get('monthly_satisfaction')
            col_f2.metric(" Μηνιαίο Satisfaction", f"{ms:.1f}%" if ms is not None else "N/A")

            if ms is not None:
                color_sat = "normal" if ms >= 65 else ("off" if ms >= 40 else "inverse")
                st.progress(int(ms))

        if 'feedback_df' in mstats:
            st.dataframe(mstats['feedback_df'], use_container_width=True, hide_index=True)

            # Pie chart μηνιαίου feedback
            if mstats['total_feedbacks'] > 0:
                cats_dict = dict(zip(
                    mstats['feedback_df']['Κατηγορία'],
                    mstats['feedback_df']['Πλήθος']
                ))
                pie_buf = chart_pie(cats_dict, mstats['total_feedbacks'],
                                    mstats.get('monthly_satisfaction', 0))
                if pie_buf:
                    col_pie1, col_pie2, col_pie3 = st.columns([1,2,1])
                    with col_pie2:
                        st.image(pie_buf, use_container_width=True)

        st.divider()

        # Συνολική Αξιολόγηση Μήνα 
        st.markdown("####  Συνολική Αξιολόγηση Μήνα")
        if 'sensor_df' in mstats:
            LIMITS_P1 = {'Θερμοκρασία (°C)':(23,27),'Υγρασία (%)':(40,60),
                         'CO2 (ppm)':(0,1000),'Θόρυβος (dBA)':(0,35)}
            LIMITS_P2 = {'Θερμοκρασία (°C)':(20,24),'Υγρασία (%)':(30,50),
                         'CO2 (ppm)':(0,1200),'Θόρυβος (dBA)':(0,35)}
            LIMITS = LIMITS_P1 if r['P1'] else LIMITS_P2

            for _, row in mstats['sensor_df'].iterrows():
                param = row['Παράμετρος']
                mean  = row['Μέσος Όρος']
                if param in LIMITS and mean != 'N/A':
                    lo, hi = LIMITS[param]
                    if lo <= mean <= hi:
                        st.success(f"✅ **{param}**: Μέσος {mean} — Εντός βέλτιστου εύρους ({lo}–{hi})")
                    else:
                        direction = "χαμηλός" if mean < lo else "υψηλός"
                        st.warning(f"⚠️ **{param}**: Μέσος {mean} — {direction.capitalize()} (βέλτιστο: {lo}–{hi})")


#  CROSS-ROOM COMPARISON
def load_room_snapshot(base_path, room, parts, sel_date, hour, minute):
    """
    Φορτώνει δεδομένα για μια αίθουσα — αν έχει πολλά parts παίρνει τον μέσο όρο.
    """
    month_num   = sel_date.strftime('%m')
    target_time = pd.Timestamp(year=sel_date.year, month=sel_date.month, day=sel_date.day,
                               hour=hour, minute=minute)
    s_start = target_time - pd.Timedelta(minutes=90)
    s_end   = target_time + pd.Timedelta(minutes=90)
    f_start = target_time - pd.Timedelta(minutes=90)
    f_end   = target_time + pd.Timedelta(minutes=90)

    all_v     = {k: [] for k in ['T','H','C','VOC','PM1','PM25','N','P']}
    all_sat   = []
    all_tf    = 0
    found_any = False

    for part in parts:
        s_path = find_sensor_path(base_path, room, part, month_num)
        f_path = find_feedback_path(base_path, room, part, month_num)

        if s_path:
            df_s = read_csv_path(s_path)
            if df_s is not None:
                df_s['Timestamp'] = pd.to_datetime(
                    df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
                snap = df_s[(df_s['Timestamp']>=s_start)&(df_s['Timestamp']<=s_end)]
                if not snap.empty:
                    found_any = True
                    m_col = next((c for c in snap.columns
                                  if 'measurement' in c.lower() and 'type' in c.lower()),
                                 next((c for c in snap.columns if 'meas' in c.lower()), None))
                    v_col = next((c for c in snap.columns if c.lower()=='value'),
                                 next((c for c in snap.columns if 'valu' in c.lower()), None))
                    if m_col and v_col:
                        def gv(names):
                            mask = snap[m_col].astype(str).str.strip().str.lower().isin(
                                [n.lower() for n in names])
                            vals = pd.to_numeric(snap.loc[mask,v_col],errors='coerce').dropna()
                            return float(vals.mean()) if not vals.empty else None
                        for key,names in [('T',['Temperature']),('H',['Humidity']),
                                          ('C',['Carbon Dioxide','CO2']),('VOC',['VOC']),
                                          ('PM1',['PM1','PM 1']),('PM25',['PM2.5','PM 2.5']),
                                          ('N',['Noise']),('P',['Pressure'])]:
                            v = gv(names)
                            if v is not None: all_v[key].append(v)

        if f_path:
            df_f = read_csv_path(f_path)
            if df_f is not None:
                df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'],utc=False,errors='coerce')
                wf = df_f[(df_f['Timestamp']>=f_start)&(df_f['Timestamp']<=f_end)]
                tf = len(wf)
                if tf > 0:
                    all_tf += tf
                    fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in wf.columns else None
                    if not fb_col:
                        fb_col = next((c for c in wf.columns
                                       if any(k in c.lower() for k in ('temp','feed','comfort'))),None)
                    if fb_col:
                        cats = wf[fb_col].astype(str).str.strip().value_counts().to_dict()
                        ws   = sum(FEEDBACK_WEIGHTS.get(l.lower(),0.0)*cnt for l,cnt in cats.items())
                        all_sat.append(ws/tf)

    # Μέσος όρος μεταξύ parts
    avg_v = {k: round(sum(vals)/len(vals),1) if vals else "N/A" for k,vals in all_v.items()}
    avg_sat = round((sum(all_sat)/len(all_sat))*100,1) if all_sat else None

    return {
        'room': room, 'v': avg_v,
        'comfort_pct': avg_sat, 'total_f': all_tf,
        'found': found_any,
    }


def chart_cross_room_bars(rooms_data, P1):
    """Bar chart σύγκρισης αιθουσών για κάθε παράμετρο."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    LIMITS = {
        'T':  (23,27) if P1 else (20,24),
        'H':  (40,60) if P1 else (30,50),
        'C':  (0,1000) if P1 else (0,1200),
        'N':  (0,35),
        'PM25':(0,12),
    }
    PARAMS = [
        ('T','Θερμοκρασία','°C'),('H','Υγρασία','%'),
        ('C','CO2','ppm'),('N','Θόρυβος','dBA'),('PM25','PM2.5','μg/m³'),
    ]
    ROOM_COLORS = {'TOFIS':'#2980B9','DRAKOS':'#E74C3C','TASOS':'#27AE60'}

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(14, 4))
    fig.patch.set_facecolor('#F8F9FA')

    for ax, (key, label, unit) in zip(axes, PARAMS):
        ax.set_facecolor('#FFFFFF')
        rooms  = [d['room'] for d in rooms_data]
        values = [d['v'].get(key) if d['v'].get(key) not in (None,"N/A") else 0
                  for d in rooms_data]
        colors = [ROOM_COLORS.get(d['room'],'#95A5A6') for d in rooms_data]

        bars = ax.bar(rooms, values, color=colors, alpha=0.82, width=0.5,
                      edgecolor='white', linewidth=0.8)
        for bar, d in zip(bars, rooms_data):
            val = d['v'].get(key)
            if val not in (None,"N/A"):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.02,
                        f'{val}', ha='center', va='bottom', fontsize=8,
                        fontweight='bold', color='#2C3E50')
            else:
                ax.text(bar.get_x()+bar.get_width()/2, 0.5,
                        'N/A', ha='center', va='bottom', fontsize=8,
                        color='#95A5A6', style='italic')

        if key in LIMITS:
            lo, hi = LIMITS[key]
            ax.axhspan(lo, hi, alpha=0.08, color='#27AE60')
            ax.axhline(y=lo, color='#27AE60', linestyle=':', linewidth=1.2, alpha=0.7)
            ax.axhline(y=hi, color='#E74C3C', linestyle=':', linewidth=1.2, alpha=0.7)
            ymax = max(max(values)*1.35, hi*1.25) if any(values) else hi*1.5
        else:
            ymax = max(values)*1.35 if any(v for v in values) else 10

        ax.set_ylim(0, max(ymax, 5))
        ax.set_title(f'{label}\n({unit})', fontsize=9, color='#2C3E50', fontweight='bold')
        ax.tick_params(axis='both', labelsize=7.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('Σύγκριση Αιθουσών — Παράμετροι IAQ', fontsize=11,
                 fontweight='bold', color='#2C3E50')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


def chart_cross_room_radar(rooms_data, P1):
    """Radar chart σύγκρισης αιθουσών."""
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    RANGES = {
        'T':   (23,27) if P1 else (20,24),
        'H':   (40,60) if P1 else (30,50),
        'C':   (400,1000) if P1 else (400,1200),
        'VOC': (0,100), 'PM25':(0,12), 'N':(0,35),
    }
    LABELS = ['Θερμοκρασία','Υγρασία','CO2','VOC','PM2.5','Θόρυβος']
    KEYS   = ['T','H','C','VOC','PM25','N']
    ROOM_COLORS = {'TOFIS':'#2980B9','DRAKOS':'#E74C3C','TASOS':'#27AE60'}

    def normalize(key, val):
        if val in (None,"N/A"): return 0.5
        lo,hi = RANGES[key]
        if key in ('C','VOC','PM25','N'):
            if val<=lo: return 1.0
            if val>=hi*1.5: return 0.0
            return max(0, 1-(val-lo)/(hi*1.5-lo))
        else:
            mid=(lo+hi)/2; span=(hi-lo)/2
            return max(0, 1-abs(val-mid)/(span*2))

    N_p = len(KEYS)
    angles = [n/float(N_p)*2*np.pi for n in range(N_p)]+[0]

    fig, ax = plt.subplots(figsize=(6,5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#F8F9FA'); ax.set_facecolor('#F0F3F4')

    for d in rooms_data:
        vals = [normalize(k, d['v'].get(k)) for k in KEYS]+[normalize(KEYS[0], d['v'].get(KEYS[0]))]
        color = ROOM_COLORS.get(d['room'],'#95A5A6')
        ax.plot(angles, vals, 'o-', linewidth=2, color=color, label=d['room'])
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(LABELS, fontsize=8.5, color='#2C3E50')
    ax.set_ylim(0,1)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], fontsize=6.5, color='#7F8C8D')
    ax.grid(color='#BDC3C7', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.set_title('IAQ Score Radar\n(1.0 = Βέλτιστο)',
                 fontsize=10, pad=18, color='#2C3E50', fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.15), fontsize=9, framealpha=0.9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close(); buf.seek(0); return buf


with tab7:
    # Σύγκριση Αιθουσών 
    st.subheader("🏫 Σύγκριση Αιθουσών — Cross-Room Analysis")
    st.caption(f"📅 {r['target_time'].strftime('%d/%m/%Y %H:%M')}  |  "
               f"{'🌤️ Περίοδος 1' if r['P1'] else '❄️ Περίοδος 2'}")
    st.info("Σύγκριση TOFIS · DRAKOS · TASOS για την ίδια ημερομηνία & ώρα. "
            "Για αίθουσες με πολλά τμήματα υπολογίζεται ο μέσος όρος.")

    with st.spinner("⏳ Φόρτωση δεδομένων όλων των αιθουσών..."):
        rooms_data = []
        for room_name, parts_list in ROOM_STRUCTURE.items():
            d = load_room_snapshot(folder_input, room_name, parts_list,
                                   sel_date, hour, minute)
            rooms_data.append(d)

    # Status table 
    st.markdown("#### 📋 Επισκόπηση")
    rows_cr = []
    for d in rooms_data:
        v = d['v']
        rows_cr.append({
            'Αίθουσα':       d['room'],
            'Δεδομένα':      '✅' if d['found'] else '❌ Δεν βρέθηκαν',
            '🌡️ T (°C)':     v.get('T','N/A'),
            '💧 H (%)':      v.get('H','N/A'),
            '💨 CO2 (ppm)':  v.get('C','N/A'),
            '🔊 Noise (dBA)':v.get('N','N/A'),
            '🌫️ PM2.5':      v.get('PM25','N/A'),
            '😊 Satisfaction':f"{d['comfort_pct']:.1f}%" if d['comfort_pct'] is not None else 'N/A',
            '📋 Feedbacks':  d['total_f'],
        })
    st.dataframe(pd.DataFrame(rows_cr), use_container_width=True, hide_index=True)

    st.divider()

    # Bar charts 
    st.markdown("#### 📊 Σύγκριση Παραμέτρων")
    found_rooms = [d for d in rooms_data if d['found']]
    if len(found_rooms) < 2:
        st.warning("⚠️ Δεν βρέθηκαν αρκετά δεδομένα για σύγκριση. "
                   "Δοκίμασε διαφορετική ώρα ή ημερομηνία.")
    else:
        st.image(chart_cross_room_bars(rooms_data, r['P1']), use_container_width=True)

        st.divider()
        st.markdown("#### 🕸️ IAQ Score Radar")
        col_r1, col_r2, col_r3 = st.columns([1,2,1])
        with col_r2:
            st.image(chart_cross_room_radar(rooms_data, r['P1']), use_container_width=True)

        st.divider()

        # Satisfaction σύγκριση 
        st.markdown("####  Satisfaction Index ανά Αίθουσα")
        sat_data_cr = [(d['room'], d['comfort_pct']) for d in rooms_data
                       if d['comfort_pct'] is not None]
        if sat_data_cr:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
            ROOM_COLORS = {'TOFIS':'#2980B9','DRAKOS':'#E74C3C','TASOS':'#27AE60'}
            fig_s, ax_s = plt.subplots(figsize=(6,3))
            fig_s.patch.set_facecolor('#F8F9FA'); ax_s.set_facecolor('#FFFFFF')
            rooms_s = [x[0] for x in sat_data_cr]
            sats_s  = [x[1] for x in sat_data_cr]
            colors_s= [ROOM_COLORS.get(r,'#95A5A6') for r in rooms_s]
            bars_s  = ax_s.bar(rooms_s, sats_s, color=colors_s, alpha=0.85,
                               width=0.5, edgecolor='white')
            for bar, sat in zip(bars_s, sats_s):
                ax_s.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                          f'{sat:.1f}%', ha='center', va='bottom',
                          fontsize=10, fontweight='bold', color='#2C3E50')
            ax_s.axhline(y=65, color='#27AE60', linestyle='--', linewidth=1.5,
                         alpha=0.7, label='Καλό (≥65%)')
            ax_s.axhline(y=40, color='#F39C12', linestyle='--', linewidth=1.5,
                         alpha=0.7, label='Μέτριο (≥40%)')
            ax_s.set_ylim(0,115)
            ax_s.set_ylabel('Satisfaction (%)', fontsize=9)
            ax_s.set_title('Student Satisfaction — Σύγκριση Αιθουσών',
                           fontsize=10, fontweight='bold', color='#2C3E50')
            ax_s.legend(fontsize=8, framealpha=0.8)
            ax_s.spines['top'].set_visible(False); ax_s.spines['right'].set_visible(False)
            plt.tight_layout()
            buf_s = io.BytesIO()
            plt.savefig(buf_s, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
            plt.close(); buf_s.seek(0)
            col_s1, col_s2, col_s3 = st.columns([1,2,1])
            with col_s2:
                st.image(buf_s, use_container_width=True)

        st.divider()

        #  Αυτόματα Συμπεράσματα 
        st.markdown("####  Αυτόματα Συμπεράσματα")
        LIMITS_CR = {
            'T':(23,27) if r['P1'] else (20,24),
            'H':(40,60) if r['P1'] else (30,50),
            'C':(0,1000) if r['P1'] else (0,1200),
            'N':(0,35),'PM25':(0,12),
        }

        def iaq_cr(d):
            scores = []
            for k,lim in LIMITS_CR.items():
                val = d['v'].get(k)
                if val not in (None,"N/A"):
                    scores.append(1 if lim[0]<=val<=lim[1] else 0)
            return sum(scores)/len(scores) if scores else 0

        scored_cr = [(d, iaq_cr(d)) for d in rooms_data if d['found']]
        if scored_cr:
            best  = max(scored_cr, key=lambda x: x[1])
            worst = min(scored_cr, key=lambda x: x[1])
            col_b, col_w = st.columns(2)
            col_b.success(f" **Καλύτερη IAQ:** {best[0]['room']} "
                          f"({best[1]*100:.0f}% εντός ορίων)")
            if worst[0]['room'] != best[0]['room']:
                col_w.error(f"⚠️ **Χειρότερη IAQ:** {worst[0]['room']} "
                            f"({worst[1]*100:.0f}% εντός ορίων)")

        sat_cr = [(d['room'], d['comfort_pct']) for d in rooms_data
                  if d['comfort_pct'] is not None]
        if len(sat_cr) >= 2:
            best_s  = max(sat_cr, key=lambda x: x[1])
            worst_s = min(sat_cr, key=lambda x: x[1])
            col_bs, col_ws = st.columns(2)
            col_bs.success(f" **Υψηλότερη Satisfaction:** {best_s[0]} ({best_s[1]:.1f}%)")
            if worst_s[0] != best_s[0]:
                col_ws.warning(f"⚠️ **Χαμηλότερη Satisfaction:** {worst_s[0]} ({worst_s[1]:.1f}%)")

    # Export PDF 
with tab8:
    st.subheader("📄 Εξαγωγή PDF Αναφοράς")
    if st.button("📥 Δημιουργία PDF", type="primary"):
        with st.spinner("Δημιουργία PDF..."):
            rtext=build_report_text(r)
            pdf_buf,err=generate_pdf(rtext,r['v'],r['comfort_pct'],
                                     r['category_counts'],r['total_f'],
                                     r['target_time'],r['season_id'])
        if err: st.error(f"❌ {err}")
        else:
            fname=f"Domognostics_{r['room']}_{r['part']}_{r['target_time'].strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button("⬇️ Κατέβασε PDF", data=pdf_buf, file_name=fname, mime="application/pdf")
            st.success("✅ Έτοιμο!")
    st.divider()
    st.subheader("📋 Προεπισκόπηση Αναφοράς")
    st.code(build_report_text(r), language=None)

