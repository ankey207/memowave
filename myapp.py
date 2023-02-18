import streamlit as st
import pytesseract
import cv2
import pandas as pd
import re
import unicodedata
import numpy as np
import os
from PIL import Image
from fuzzywuzzy import process
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
# Page setting
st.set_page_config(page_title="MemoWAVE", page_icon="favicon.png", layout="wide", initial_sidebar_state="auto", menu_items=None)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


#supprimer tous les accents
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

#supprimer les differents texte avant les transactions
def remove_error2(liste, keywords):
    # Trouver l'index de la première ligne qui contient l'un des mots clés
    indexs = []
    for element in liste:
        if element.split()[0] in keywords:
            indexs.append(liste.index(element))

    # Si au moins un des mots clés est présent dans le texte
    if len(indexs) != 0:
        # Garder seulement les lignes suivantes la première occurrence d'un mot clé
        filtered_lines = liste[min(indexs):]
    else:
        # Garder toutes les lignes si aucun mot clé n'est présent
        filtered_lines = liste
    
    return filtered_lines


def remove_error(text):
    indexs=[]
    PRENOMSs=['De','Depot','Depet','A','Retrait','Paiement','Withdrawal','Received','Sent','Transfer','Deposit','Paid']
    for fisrt_word in PRENOMSs:
        if fisrt_word in text:
            indexs.append(text.index(fisrt_word))
    if indexs:
        text = text[min(indexs):]
    return text

#supprimer les virgules et les points dans la liste de date
def delete_coma(var:list):
    for i in range(len(var)):
        var[i] = var[i].replace(".", " ")
        var[i] = var[i].replace(",", " ")
    return var


        #ppp
def remove_prenoms_from_list(liste:list):
    PRENOMS=['De','Depot','Depet','A','Retrait','Paiement','Payé','Bonus','Transfert','Withdrawal','Received','Sent','Transfer','Deposit','Paid']
    new_liste=liste
    liste_prenoms=[]
    elements_to_remove = []
    for elment in liste:
        if elment.split()[0] in PRENOMS :
            elements_to_remove.append(elment)
            liste_prenoms.append(elment)
        else:
            var = elment.split()[0]
            if var[0] in PRENOMS:
                elements_to_remove.append(elment)
                liste_prenoms.append(elment)

    for element in elements_to_remove:
        new_liste.remove(element)
    return [new_liste, liste_prenoms]


def remove_montant_from_list(liste:list):
    new_liste=liste
    liste_montant=[]
    elements_to_remove = []
    for elment in liste:
        if elment[-1] == 'F' :
            elements_to_remove.append(elment)
            liste_montant.append(elment)
    for element in elements_to_remove:
        new_liste.remove(element)
    return [new_liste, liste_montant]


def delete_last_element(liste1, liste2, liste3,liste4):
    taille_min = min(len(liste1), len(liste2), len(liste3))
    while len(liste1) > taille_min:
        liste1.pop()
    while len(liste2) > taille_min:
        liste2.pop()
    while len(liste3) > taille_min:
        liste3.pop()
    while len(liste4) > taille_min:
        liste4.pop()

def delete_last_element_11(liste1, liste2, liste3):
    taille_min = min(len(liste1), len(liste2), len(liste3))
    while len(liste1) > taille_min:
        liste1.pop()
    while len(liste2) > taille_min:
        liste2.pop()
    while len(liste3) > taille_min:
        liste3.pop()


#corriger de les mois des differentes dates
def right_month(chaine):
    MOIS = ['janv', 'fevr', 'mars', 'avr', 'mai', 'juin', 'juil', 'aout', 'sept', 'oct', 'nov', 'dec']
    resultat = process.extractOne(chaine, MOIS)
    if resultat[1] >= 50:
        resultat = resultat[0]
    else:
        resultat = 'None'

    return resultat

#supprimer les numeros ou les autres elements qui ne doiebnt pas etre dans la liste de date
def delete_intrus_in_date(var:list):
    new_var =  var
    elements_to_remove = []
    for elment in var:
        if right_month(elment.split()[0])=='None'  or len(elment)<4:
            elements_to_remove.append(elment)
    for elment in elements_to_remove:
        new_var.remove(elment)
    return new_var

def delete_intrus_in_date_english(var:list):
    new_var =  var
    elements_to_remove = []
    for elment in var:
        if len(elment.split()) == 1:
            elements_to_remove.append(elment)

        elif len(elment.split()) == 2:
            if right_month(elment.split()[1])=='None':
                elements_to_remove.append(elment)

        else:
            if right_month(elment.split()[1])=='None':
                elements_to_remove.append(elment)
    for elment in elements_to_remove:
        new_var.remove(elment)
    return new_var


def get_TRANSACTION(word):
    if word == 'De' or word == 'Depot' or word == 'Depet' or word =='Bonus' or word =='Deposit' or word =='Received':
        return 'Depot'
    if word == 'A' or word == 'Transfert' or word == 'Sent' or word == 'Transfer':
        return 'Transfert'
    if word == 'Retrait' or word == 'Withdrawal':
        return 'Retrait'
    else:
        return 'Paiement'

def get_name(var):
    text=var.split()
    if text[0] == 'Depot' or text[0] == 'Depet' or text[0] == 'Retrait' or text[0] == 'Withdrawal' or text[0] == 'Deposit':
        return "Moi"
    elif text[0] == 'Bonus':
        return 'Bonus'
    else:
        pattern = r'^\D+'
        match = re.search(pattern, var)
        if match:
            text=match.group()
            return text[text.find(" "):].strip()
        else:
            return "NOT FOUND"

#fonction qui permet de retrouver le numero du mois
def true_month(var):
    if var == "janv":
        return 1
    if var == "fevr":
        return 2
    if var == "mars":
        return 3
    if var == "avr":
        return 4
    if var == "mai":
        return 5
    if var == "juin":
        return 6
    if var == "juil":
        return 7
    if var == "aout":
        return 8
    if var == "sept":
        return 9
    if var == "oct":
        return 10
    if var == "nov":
        return 11
    if var == "dec":
        return 12

#fonction permettant d'obtenir la date dans le bon format
def get_date(var):
    text=var.split()
    if re.search("[a-zA-Z]", text[1]):
        return "error"
    else:
        month=right_month(text[0])
        day=re.sub(r"^0+", "", text[1])

        if len(text)==2:
            year=datetime.now().year
            #year=2022

        if len(text)>2:
            year=text[2]
            if ':' in year:
                year=datetime.now().year

            if len(str(year))!=4:
                year=datetime.now().year

            if len(str(year)) ==4 and (int(year) < 2018 or int(year) > datetime.now().year):
                year=datetime.now().year
        try:
            date = datetime(int(year), true_month(month),int(day))
            formatted_date = date.strftime("%Y-%m-%d")
        except:
            formatted_date = "error"
        return formatted_date



def get_date_english(var):
    text=var.split()
    month=right_month(text[1])
    day= text[0]

    if len(text)==2:
        year=datetime.now().year
        #year=2022

    if len(text)>2:
        year=text[2]
        if ':' in year:
            year=datetime.now().year

        if len(year)!=4:
            year=datetime.now().year

        if len(year) ==4 and (int(year) < 2018 or int(year) > datetime.now().year):
            year=datetime.now().year

    try:
        date = datetime(int(year), true_month(month),int(day))
        formatted_date = date.strftime("%Y-%m-%d")
    except:
        formatted_date = "error"

    return formatted_date

def corriger_montant(chaine):
    # Table de correspondance
    correspondance = {
        "Z": "2",
        "S": "5",
        "O": "0",
        "A": "4",
        "T": "1",
        "I": "1",
        "G": "6",
        "H": "4",
        "H": "4",
        "~": "-",
        "_": "-",
        "“": "-",
        '"': "-",
        "..": "-",
        '``': "-"
        
    }
    # Parcours de la chaîne de caractères et remplacement des caractères incorrects
    for i in range(len(chaine)):
        chaine = chaine.replace(".", "")
        chaine = chaine.replace("F", "")
        chaine = chaine.replace(",", "")
        chaine = chaine.replace(":", "")
        chaine = chaine.replace("!", "")
        chaine = chaine.replace("?", "")
        chaine = chaine.replace("|", "")


    for i in range(len(chaine)):
        if chaine[i] in correspondance:
            chaine = chaine[:i] + correspondance[chaine[i]] + chaine[i+1:]

    if re.search(r'[a-zA-Z]', chaine):
        return "Erreur"
    else:
        chaine =abs(int(chaine))
        return chaine

def get_language(prenom:list,date:list):
    fra = 0
    eng = 0
    for i in range(min(len(prenom),len(date))):
        x = prenom[i]
        y = date[i]
        if 'Withdrawal' in x or 'Received' in x or 'Sent' in x or 'Deposit' in x or 'Paid' in x:
            if 'Jan ' in y or 'Feb ' in y or 'Mar ' in y or 'Apr ' in y or 'Jun ' in y or 'Jul ' in y or 'Aug ' in y or 'Sep ' in y or 'Nov ' in y or 'Dec ' in y:
                 fra +=1
            else:
                eng +=1
        else:
            fra +=1
    if eng != 0:
        return 'eng'
    else:
        return 'fra'


def corriger_prenom_english(chaine:list):
    chaine = chaine.replace("to ", "")
    chaine = chaine.replace("from ", "")
    return chaine

def extract_characters_before_first_digit(var:list):
    new_liste = []
    for word in var:
        # Utilise la regex pour trouver le premier chiffre dans la chaîne
        match = re.search(r'\d', word)
        if match:
            # Si un chiffre est trouvé, retourne tous les caractères précédents
            new_liste.append(word[:match.start()])
        else:
            # Si aucun chiffre n'est trouvé, retourne la chaîne complète
            new_liste.append(word)
    return new_liste

def get_all_numbers(var:list):
    new_liste = []
    regex = r"\d{2}\s*\d{2}\s*\d{2}\s*\d{2}\s*\d{2}"
    for word in var:
        match = re.search(regex, word)
        if match:
            new_liste.append( "".join(match.group().split()))
        else:
            new_liste.append('NOT FOUND')
    return new_liste

def get_all_montant(var:list):
    new_liste = []
    for word in var:
        word = word.strip()
        word = word.split()
        new_liste.append(word[-1])
    return new_liste

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

#charger les images
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)


st.image(Image.open('memowave.png'),width=300)
z1, z2 = st.columns(2)
with z1:
    st.title(":blue[Choisissez une ou plusieurs images:]")
with z2:
    uploaded_files = st.file_uploader(":blue[Choisissez une ou plusieurs images:]",accept_multiple_files=True,type=["png", "jpg", "jpeg"],label_visibility='hidden')
try:
    if uploaded_files is not None:
        PRENOMS_NUM=[]
        DATE=[]
        MONTANT=[]
        NUMEROS = []
        NOMBRE_LIGNE_IMAGE = []
        config = r'--psm 1'


        config_11 = r'--psm 11'


        for file in uploaded_files:
            lISTE=[]
            lISTE_11 = []
            img_cv = Image.open(file)
            img_cv = img_cv.save("img.jpg")
            img_cv1 = cv2.imread("img.jpg")
            img_cv1 = unsharp_mask(img_cv1)
            texte = pytesseract.image_to_string(img_cv1,config=config)
            texte_11 = pytesseract.image_to_string(img_cv1,config=config_11)

            texte = remove_accents(texte)
            texte = texte.strip()
            

            with open("text.txt", "w") as file1:
              file1.write(texte)

            with open("text.txt", "r") as f:
                for line in f:
                    lISTE.append(line.strip())


            def delete_empty_value(var):
                return [x for x in var if x]

            lISTE =delete_empty_value(lISTE)
            keywords = ['De','Depot','Depet','A','Retrait','Paiement','Withdrawal','Received','Sent','Transfer','Deposit','Paid']
            lISTE = remove_error2(lISTE, keywords)

            lISTE = remove_prenoms_from_list(lISTE)
            LISTE_PRENOMS_NUMs = lISTE[1]
            LISTE_PRENOMS_NUM = extract_characters_before_first_digit(LISTE_PRENOMS_NUMs)
            lISTE_NUMEROS = get_all_numbers(LISTE_PRENOMS_NUMs)
            lISTE_MONTANT = get_all_montant(LISTE_PRENOMS_NUMs)
            

            #la liste avec uniquement les date et premeir traitrement des dates
            lISTE_DATE = lISTE[0]
            lISTE_DATE = delete_coma(lISTE_DATE)

            lang = get_language(LISTE_PRENOMS_NUM,lISTE_DATE)            
            if lang == 'eng':
                lISTE_DATE = delete_intrus_in_date_english(lISTE_DATE)
            if lang == 'fra':
                lISTE_DATE = delete_intrus_in_date(lISTE_DATE)

            #recuperation de la liste des monatnt

            #un peu d'appurement des liste supprimer les residus
            delete_last_element(LISTE_PRENOMS_NUM,lISTE_DATE,lISTE_MONTANT,lISTE_NUMEROS)


            #on recupere le nombre de lignes dans chaque images
            NOMBRE_LIGNE_IMAGE.append(len(LISTE_PRENOMS_NUM))
            
            #conception de liste unique pour tous les images selectionnées
            PRENOMS_NUM += LISTE_PRENOMS_NUM
            DATE += lISTE_DATE
            MONTANT += lISTE_MONTANT
            NUMEROS += lISTE_NUMEROS

        st.write("1")
        st.write(PRENOMS_NUM)
        st.write(DATE)
        st.write(MONTANT)
        st.write(NUMEROS)

        data = pd.DataFrame(list(zip(PRENOMS_NUM, DATE, MONTANT,NUMEROS)), columns=['TEXT', 'DATE1', 'MONATANT1','NUMEROS'])

        data['PRENOMS'] = data['TEXT'].str.split(' ').str[0]
        data['TRANSACTION'] = data['PRENOMS'].apply(get_TRANSACTION)
        data['PRENOMS'] = data['TEXT'].apply(get_name)
        data['PRENOMS'] = data['PRENOMS'].apply(corriger_prenom_english)
        data['MONTANT'] = data['MONATANT1'].apply(corriger_montant)
        
        data['DATE'] = data['DATE1']
        lang = get_language(PRENOMS_NUM,DATE)
        if lang == 'eng':
            data['DATE'] = data['DATE1'].apply(get_date_english)
        if lang == 'fra':
            data['DATE'] = data['DATE1'].apply(get_date)
        data = data.loc[data['DATE'] != 'error']

        #consolidation1 de la base finale
        data = data[['PRENOMS','TRANSACTION','DATE','MONTANT','NUMEROS']]

        #data['DATE1'] = pd.to_datetime(data.loc[:,'DATE'], format='%Y-%m-%d')
        #data = data.sort_values(by='DATE1',ascending=False)
        #data = data[['PRENOMS','TRANSACTION','DATE','MONTANT']]

        # Calculer la somme des montants de retrait
        data_retraits = data.loc[data['TRANSACTION'] == 'Retrait']
        somme_des_retraits = data_retraits['MONTANT'].sum()

        # Calculer la somme des montants de depot
        data_depot = data.loc[data['TRANSACTION'] == 'Depot']
        somme_des_depots = data_depot['MONTANT'].sum()

        # Calculer la somme des montants des paiement en ligne
        data_paiement = data.loc[data['TRANSACTION'] == 'Paiement']
        somme_des_paiements = data_paiement['MONTANT'].sum()

        # Calculer la somme des montants transféré
        data_transfert = data.loc[data['TRANSACTION'] == 'Transfert']
        somme_Transfert = data_transfert['MONTANT'].sum()

        # la data des depenses
        data_depenses = data.loc[data['TRANSACTION'] != 'Depot']

        # intervalle des temps des transactions
        period_min = data['DATE'].min()
        period_max = data['DATE'].max()

        #ligne momo wave
        
        #st.write(":blue[statistiques pour la période du ]"+str(period_min))
        #st.success('This is a success message!', icon="✅")
        if len(data)==0:
            st.info("Ce site vous permet de visualiser des statistiques de vos transactions WAVE en utilisant des captures d'écran.", icon="ℹ️")
            st.warning("Les images ne doivent comporter aucun élément extérieur à l'application WAVE, sous peine de voir certaines informations ne pas être prises en compte. ", icon="⚠️")
        if len(data) != 0:
            st.markdown(f"<h4 style='color: #0068c9;text-align: center; background-color: #eeeeee';text-shadow: 2px 2px;>MemoWAVE pour la période du {period_min} au {period_max} </h4>", unsafe_allow_html=True)
            st.markdown("")
            
        # Ligne A
        
        #if len(data)!=0 and isinstance(somme_des_retraits, int) and isinstance(somme_des_depots, int) and isinstance(somme_Transfert, int):
        if len(data)!=0:
            a1, a2, a3 = st.columns(3,gap="small")
            a1.metric(":blue[Montant Total rétiré:]", "{:,.0f} CFA".format(somme_des_retraits))
            a2.metric(":blue[Montant Total déposé:] ", "{:,.0f} CFA".format(somme_des_depots))
            a3.metric(":blue[Montant Total transféré:] ", "{:,.0f} CFA".format(somme_Transfert))

        # Ligne B
        b1, b2 = st.columns(2)
        with b1:
            #courbe d'evolution des depenses sur le temps jour
            courbe = data_depenses.groupby('DATE').sum(numeric_only=True)
            if len(courbe)!=0:
                courbe = pd.DataFrame(list(zip(courbe['MONTANT'],courbe['MONTANT'].index)), columns=['MONTANT','DATE' ])
                courbe['DATE'] = pd.to_datetime(courbe.loc[:,'DATE'],format='%Y-%m-%d')
                courbe = courbe.sort_values(by='DATE',ascending=True)
                #st.write(courbe)
                #courbe d'evolution
                fig_courbe = px.line(courbe,x="DATE",y="MONTANT", title="Evolution des Dépenses sur le temps")
                fig_courbe.update_traces(hovertemplate='<b>Date : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')
                fig_courbe.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
                fig_courbe.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
                st.plotly_chart(fig_courbe, theme="streamlit", use_container_width=True)

        with b2:
            # diagramme en baton top 5 des tranfert
            bar_transfert = data_transfert.groupby('PRENOMS').sum(numeric_only=True)
            if len(bar_transfert)!=0:
                bar_transfert = pd.DataFrame(list(zip(bar_transfert['MONTANT'],bar_transfert['MONTANT'].index)), columns=['MONTANT','PRENOMS' ])
                bar_transfert = bar_transfert.sort_values(by='MONTANT',ascending=False)
                bar_transfert= bar_transfert[:5]
                #st.write(bar_transfert)
                #courbe d'evolution
                fig_bar_transfert = px.bar(bar_transfert, y='MONTANT', x='PRENOMS', text_auto='.2s',title="TOP5 bénéficiaires de vos Transferts")
                fig_bar_transfert.update_traces(hovertemplate='<b>Prénoms : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')
                fig_bar_transfert.update_traces(texttemplate='%{x} ',textfont_size=16, textangle=0, textposition="inside", cliponaxis=False)
                fig_bar_transfert.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
                fig_bar_transfert.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
                st.plotly_chart(fig_bar_transfert, theme="streamlit", use_container_width=True)

        # Ligne C
        c1, c2 = st.columns((12,5))

        with c1:
            #histogramme volume des depenses groupé par mois
            histo = data_depenses.copy()
            histo['DATE'] = pd.to_datetime(histo.loc[:,'DATE'], format='%Y-%m-%d')
            histo['DATE'] = histo.loc[:,'DATE'].dt.strftime('%b %Y')
            histo = histo.groupby('DATE').sum(numeric_only=True)
            if len(histo)!=0:
                histo = pd.DataFrame(list(zip(histo['MONTANT'],histo['MONTANT'].index)), columns=['MONTANT','DATE' ])
                histo['DATE'] = pd.to_datetime(histo['DATE'])
                histo = histo.sort_values(by='DATE',ascending=True)
                #st.write(histo)
                fig_histo = px.bar(histo, x="DATE",y="MONTANT", title=" Volume de vos Dépenses par mois")
                fig_histo.update_traces(hovertemplate='<b>Date : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')
                fig_histo.update_traces(texttemplate='%{y} CFA', textposition='inside')
                fig_histo.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
                fig_histo.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
                st.plotly_chart(fig_histo, theme="streamlit", use_container_width=True)

        with c2:   

            @st.experimental_memo
            def convert_df(data):
               return data.to_csv(index=False).encode('utf-8')
            csv = convert_df(data)
            if len(histo)!=0:
                st.download_button(
                   "Télécharger les données en format CSV",
                   csv,
                   "file.csv",
                   "text/csv",
                   key='download-csv'
                )
            #integralité des transactions WAVE
            #df=data
            if len(histo)!=0:
                st.dataframe(data, width=None, height=None)


        # diagramme en baton top 5 des depots
        pie_depot = data_depot.groupby('PRENOMS').sum(numeric_only=True)
        if len(pie_depot)!=0:
            pie_depot = pd.DataFrame(list(zip(pie_depot['MONTANT'],pie_depot['MONTANT'].index)), columns=['MONTANT','PRENOMS' ])
            pie_depot = pie_depot.sort_values(by='MONTANT',ascending=False)
            pie_depot= pie_depot[:5]
            #st.write(pie_depot)
            #courbe d'evolution
            fig_bar_transfert = px.bar(pie_depot, y='MONTANT', x='PRENOMS', text_auto='.2s',title="TOP5 Donateurs ")
            fig_bar_transfert.update_traces(hovertemplate='<b>Prénoms : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')

            fig_bar_transfert.update_traces(texttemplate='%{y} CFA',textfont_size=16, textangle=0, textposition="inside", cliponaxis=False)
            fig_bar_transfert.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
            fig_bar_transfert.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
            st.plotly_chart(fig_bar_transfert, theme="streamlit", use_container_width=True)
except:
    try:
        PRENOMS_NUM=[]
        DATE=[]
        MONTANT=[]
        NUMEROS = []
        NOMBRE_LIGNE_IMAGE = []
        config_11 = r'--psm 11'

        for file in uploaded_files:
            lISTE=[]
            img_cv = Image.open(file)
            img_cv = img_cv.save("img.jpg")
            img_cv1 = cv2.imread("img.jpg")
            img_cv1 = unsharp_mask(img_cv1)
            texte = pytesseract.image_to_string(img_cv1,config=config_11)
            texte = remove_accents(texte)

            texte = texte.strip()
            with open("text1.txt", "w") as file1:
              file1.write(texte)

            with open("text1.txt", "r") as f:
                for line in f:
                    lISTE.append(line.strip())

            def delete_empty_value(var):
                return [x for x in var if x]

            lISTE =delete_empty_value(lISTE)
            keywords = ['De','Depot','Depet','A','Retrait','Paiement','Withdrawal','Received','Sent','Transfer','Deposit','Paid']
            lISTE = remove_error2(lISTE, keywords)
            lISTE = remove_prenoms_from_list(lISTE)
            LISTE_PRENOMS_NUM = lISTE[1]
            lISTE_NUMEROS = get_all_numbers(LISTE_PRENOMS_NUM)

            #la liste sans les prenoms et les numeros
            lISTE = lISTE[0]

            #la liste avec uniquement les date et premeir traitrement des dates
            lISTE = remove_montant_from_list(lISTE)
            lISTE_DATE = lISTE[0]
            lISTE_DATE = delete_coma(lISTE_DATE)

            lang = get_language(LISTE_PRENOMS_NUM,lISTE_DATE)
            if lang == 'eng':
                lISTE_DATE = delete_intrus_in_date_english(lISTE_DATE)
            if lang == 'fra':
                lISTE_DATE = delete_intrus_in_date(lISTE_DATE)



            #recuperation de la liste des monatnt
            lISTE_MONTANT = lISTE[1]

            #un peu d'appurement des liste supprimer les residus
            delete_last_element_11(LISTE_PRENOMS_NUM,lISTE_DATE,lISTE_MONTANT)

            #on recupere le nombre de lignes dans chaque images
            NOMBRE_LIGNE_IMAGE.append(len(LISTE_PRENOMS_NUM))
            
            #conception de liste unique pour tous les images selectionnées
            PRENOMS_NUM += LISTE_PRENOMS_NUM
            DATE += lISTE_DATE
            MONTANT += lISTE_MONTANT
            NUMEROS += lISTE_NUMEROS

        st.write("2")
        st.write(PRENOMS_NUM)
        st.write(DATE)
        st.write(MONTANT)
        st.write(NUMEROS)

        data = pd.DataFrame(list(zip(PRENOMS_NUM, DATE, MONTANT,NUMEROS)), columns=['TEXT', 'DATE1', 'MONATANT1','NUMEROS'])
        data['PRENOMS'] = data['TEXT'].str.split(' ').str[0]
        data['TRANSACTION'] = data['PRENOMS'].apply(get_TRANSACTION)
        data['PRENOMS'] = data['TEXT'].apply(get_name)
        data['PRENOMS'] = data['PRENOMS'].apply(corriger_prenom_english)
        data['MONTANT'] = data['MONATANT1'].apply(corriger_montant)

        data['DATE'] = data['DATE1']
        lang = get_language(PRENOMS_NUM,DATE)
        if lang == 'eng':
            data['DATE'] = data['DATE1'].apply(get_date_english)
        if lang == 'fra':
            data['DATE'] = data['DATE1'].apply(get_date)

        data = data.loc[data['DATE'] != 'error']


        #consolidation1 de la base finale
        #data = data[['PRENOMS','TRANSACTION','DATE','MONTANT']]
        #data['DATE1'] = pd.to_datetime(data.loc[:,'DATE'], format='%Y-%m-%d')
        #data = data.sort_values(by='DATE1',ascending=False)
        data = data[['PRENOMS','TRANSACTION','DATE','MONTANT','NUMEROS']]

        # Calculer la somme des montants de retrait
        data_retraits = data.loc[data['TRANSACTION'] == 'Retrait']
        somme_des_retraits = data_retraits['MONTANT'].sum()

        # Calculer la somme des montants de depot
        data_depot = data.loc[data['TRANSACTION'] == 'Depot']
        somme_des_depots = data_depot['MONTANT'].sum()

        # Calculer la somme des montants des paiement en ligne
        data_paiement = data.loc[data['TRANSACTION'] == 'Paiement']
        somme_des_paiements = data_paiement['MONTANT'].sum()

        # Calculer la somme des montants transféré
        data_transfert = data.loc[data['TRANSACTION'] == 'Transfert']
        somme_Transfert = data_transfert['MONTANT'].sum()

        # la data des depenses
        data_depenses = data.loc[data['TRANSACTION'] != 'Depot']

        # intervalle des temps des transactions
        period_min = data['DATE'].min()
        period_max = data['DATE'].max()


        #ligne momo wave
        
        #st.write(":blue[statistiques pour la période du ]"+str(period_min))
        #st.success('This is a success message!', icon="✅")
        if len(data)==0:
            st.info("Ce site vous permet de visualiser des statistiques de vos transactions WAVE en utilisant des captures d'écran.", icon="ℹ️")
            st.warning("Les images ne doivent comporter aucun élément extérieur à l'application WAVE, sous peine de voir certaines informations ne pas être prises en compte. ", icon="⚠️")
        if len(data) != 0:
            st.markdown(f"<h4 style='color: #0068c9;text-align: center; background-color: #eeeeee';text-shadow: 2px 2px;>MemoWAVE pour la période du {period_min} au {period_max} </h4>", unsafe_allow_html=True)
            st.markdown("")
            
        # Ligne A
        
        #if len(data)!=0:
        a1, a2, a3 = st.columns(3,gap="small")
        a1.metric(":blue[Montant Total rétiré:]", "{:,.0f} CFA".format(somme_des_retraits))
        a2.metric(":blue[Montant Total déposé:] ", "{:,.0f} CFA".format(somme_des_depots))
        a3.metric(":blue[Montant Total transféré:] ", "{:,.0f} CFA".format(somme_Transfert))

        # Ligne B
        b1, b2 = st.columns(2)
        with b1:
            #courbe d'evolution des depenses sur le temps jour
            courbe = data_depenses.groupby('DATE').sum(numeric_only=True)
            if len(courbe)!=0:
                courbe = pd.DataFrame(list(zip(courbe['MONTANT'],courbe['MONTANT'].index)), columns=['MONTANT','DATE' ])
                courbe['DATE'] = pd.to_datetime(courbe.loc[:,'DATE'],format='%Y-%m-%d')
                courbe = courbe.sort_values(by='DATE',ascending=True)
                #st.write(courbe)
                #courbe d'evolution
                fig_courbe = px.line(courbe,x="DATE",y="MONTANT", title="Evolution des Dépenses sur le temps")
                fig_courbe.update_traces(hovertemplate='<b>Date : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')
                fig_courbe.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
                fig_courbe.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
                st.plotly_chart(fig_courbe, theme="streamlit", use_container_width=True)

        with b2:
            # diagramme en baton top 5 des tranfert
            bar_transfert = data_transfert.groupby('PRENOMS').sum(numeric_only=True)
            if len(bar_transfert)!=0:
                bar_transfert = pd.DataFrame(list(zip(bar_transfert['MONTANT'],bar_transfert['MONTANT'].index)), columns=['MONTANT','PRENOMS' ])
                bar_transfert = bar_transfert.sort_values(by='MONTANT',ascending=False)
                bar_transfert= bar_transfert[:5]
                #st.write(bar_transfert)
                #courbe d'evolution
                fig_bar_transfert = px.bar(bar_transfert, y='MONTANT', x='PRENOMS', text_auto='.2s',title="TOP5 bénéficiaires de vos Transferts")
                fig_bar_transfert.update_traces(hovertemplate='<b>Prénoms : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')
                fig_bar_transfert.update_traces(texttemplate='%{x} ',textfont_size=16, textangle=0, textposition="inside", cliponaxis=False)
                fig_bar_transfert.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
                fig_bar_transfert.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
                st.plotly_chart(fig_bar_transfert, theme="streamlit", use_container_width=True)

        # Ligne C
        c1, c2 = st.columns((12,5))

        with c1:
            #histogramme volume des depenses groupé par mois
            histo = data_depenses.copy()
            histo['DATE'] = pd.to_datetime(histo.loc[:,'DATE'], format='%Y-%m-%d')
            histo['DATE'] = histo.loc[:,'DATE'].dt.strftime('%b %Y')
            histo = histo.groupby('DATE').sum(numeric_only=True)
            if len(histo)!=0:
                histo = pd.DataFrame(list(zip(histo['MONTANT'],histo['MONTANT'].index)), columns=['MONTANT','DATE' ])
                histo['DATE'] = pd.to_datetime(histo['DATE'])
                histo = histo.sort_values(by='DATE',ascending=True)
                #st.write(histo)
                fig_histo = px.bar(histo, x="DATE",y="MONTANT", title=" Volume de vos Dépenses par mois")
                fig_histo.update_traces(hovertemplate='<b>Date : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')
                fig_histo.update_traces(texttemplate='%{y} CFA', textposition='inside')
                fig_histo.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
                fig_histo.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
                st.plotly_chart(fig_histo, theme="streamlit", use_container_width=True)

        with c2:   

            @st.experimental_memo
            def convert_df(data):
               return data.to_csv(index=False).encode('utf-8')
            csv = convert_df(data)
            if len(histo)!=0:
                st.download_button(
                   "Télécharger les données en format CSV",
                   csv,
                   "file.csv",
                   "text/csv",
                   key='download-csv'
                )
            #integralité des transactions WAVE
            df=data
            if len(histo)!=0:
                st.dataframe(df, width=None, height=None)


        # diagramme en baton top 5 des depots
        pie_depot = data_depot.groupby('PRENOMS').sum(numeric_only=True)
        if len(pie_depot)!=0:
            pie_depot = pd.DataFrame(list(zip(pie_depot['MONTANT'],pie_depot['MONTANT'].index)), columns=['MONTANT','PRENOMS' ])
            pie_depot = pie_depot.sort_values(by='MONTANT',ascending=False)
            pie_depot= pie_depot[:5]
            #st.write(pie_depot)
            #courbe d'evolution
            fig_bar_transfert = px.bar(pie_depot, y='MONTANT', x='PRENOMS', text_auto='.2s',title="TOP5 Donateurs ",height=400,)
            fig_bar_transfert.update_traces(hovertemplate='<b>Prénoms : </b> %{x} <br>' + '<b>Montant Total : </b> %{y} CFA')

            fig_bar_transfert.update_traces(texttemplate='%{y} CFA',textfont_size=16, textangle=0, textposition="inside", cliponaxis=False)
            fig_bar_transfert.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
            fig_bar_transfert.update_layout(xaxis={'fixedrange':True},yaxis={'fixedrange':True})
            st.plotly_chart(fig_bar_transfert, theme="streamlit", use_container_width=True)

    except:
        st.warning("Il semble que l'une ou plusieurs de vos images ne sont pas conformes. S'il vous plaît, veuillez réessayer. ", icon="⚠️")


footer="""<style>
    a:link , a:visited{
    color: blue;
    background-color: transparent;
    text-decoration: underline;
    }

    a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
    }

    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
</style>
<div class="footer">
    <p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/nsi%C3%A9ni-kouadio-eli%C3%A9zer-amany-613681185" target="_blank">Nsiéni Amany Kouadio</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)