import pandas as pd
import numpy as np
import datetime as dt
import pickle
import inflection
import math



class Rossmann (object):
    def __init__(self):
        self.home_path = 'C:/Users/Jacqueline/Repos/DsEmPoducao/'
        self.competition_distance_scaler   = pickle.load (open( self.home_path + 'parameter/competition_distance_scaler.pkl' ,'rb'))
        self.competition_time_month_scaler = pickle.load (open( self.home_path + 'parameter/competition_time_month_scaler.pkl','rb' ))
        self.promo_time_week_scaler        = pickle.load (open( self.home_path + 'parameter/promo_time_week_scaler.pkl' ,'rb' ))
        self.year_scaler                   = pickle.load (open( self.home_path + 'parameter/year_scaler.pkl','rb'))
        self.store_type_scaler             = pickle.load (open( self.home_path + 'parameter/store_type_scaler.pkl','rb'))



 
    def data_cleaning ( self,df1 ):
        ##1.1 Rename Columns
        cols_old = ['Store', 'DayOfWeek', 'Date',  'Open', 'Promo','StateHoliday',                                             'SchoolHoliday','StoreType','Assortment','CompetitionDistance',                                                                 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear',                                                                         'Promo2','Promo2SinceWeek','Promo2SinceYear', 'PromoInterval']
        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase,cols_old))
        df1.columns = cols_new
        ##1.3 Data Types
        #Tranformando o campo data em tipo data
        df1['date'] = pd.to_datetime(df1['date'])
    
        ## 1.5 Fillout NA
    
        #competition_distance             
        df1['competition_distance'] =df1['competition_distance'].apply(lambda x: 200000 if math.isnan(x) else x)
    
        #competition_open_since_month    
        df1['competition_open_since_month']= df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month'])       else x['competition_open_since_month'], axis = 1)
    
        #competition_open_since_year     
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year  if math.isnan(x['competition_open_since_year'])         else x['competition_open_since_year'], axis = 1 )
    
        #promo2_since_week        
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else                    x['promo2_since_week'], axis = 1)
    
        #promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x : x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis = 1)
    
        #promo_interval            
        month_map = {1  : 'Jan',2  : 'Feb',3  : 'Mar',4  : 'Apr',5  : 'May',6  : 'Jun',7  : 'Jul',8  : 'Aug',9  : 'Sep',10 : 'Oct',11 : 'Nov',12 : 'Dec' }
    
        df1['promo_interval'].fillna(0,inplace = True)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
    
    
        #comparando se o mes de venda 'date'está contido no periodo de promocao 'promo_intervale separando os meses da coluna promo_interval com espaco pois estao todos juntos'
        df1['is_promo'] =df1[['promo_interval', 'month_map']].apply                                                                            (lambda x: 0 if x['promo_interval'] == 0 else                                                                                             1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis= 1)
        ##1.6 Change Types
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
        df1['promo2_since_week']= df1['promo2_since_week'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
    
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
        df1['promo2_since_week']= df1['promo2_since_week'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
    
        return df1

    def feature_engineering ( self,df2):
        #year
        df2['year'] = df2['date'].dt.year

        #month
        df2['month'] = df2['date'].dt.month

        #day
        df2['day']= df2['date'].dt.day

        #week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        #year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')


        #competion since
        df2['competition_since']=df2.apply(lambda x: dt.datetime(year = x['competition_open_since_year'], month = x['competition_open_since_month'], day = 1), axis=1)

        df2['competition_time_month']=((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype('int64')
        #promo since
        ##convertendo o ano de float para inteiro
        df2['promo2_since_year'] = df2['promo2_since_year'].astype('int64')

        ##juntando o ano com o mes
        df2['promo_since'] = df2['promo2_since_year'].astype(str) +  '-' + df2['promo2_since_week'].astype(str)

        ##convertendo a data em semana
        df2['promo_since'] = df2['promo_since'].apply (lambda x: dt.datetime.strptime (x+ '-1', '%Y-%W-%w') - dt.timedelta(days=7))

        ##subraindo a data de promocao da data de venda para achar quantas semanas ficou em promocao
        df2['promo_time_week']= ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype('int64')


        #assortment
        ##atribuindo o significado de cada letra no dataframe
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x== 'a' else 'extra' if x== 'b' else 'extend')

        #state holiday
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x=='b' else 'Christmas' if x == 'c' else 'regular_day')

        ## 3.1 Filtragem das linhas
        #Filtrando as linhas de lojas fechdas pois nao faz sentindo fazer uma previsao de venda de loja fechada e tbm de vendas diferentes de zero

        #Filtragem de linhas a serem utilizadas
        df2 = df2[df2['open'] !=0]
   
        #4.2  3.2 Seleção das colunas
        #Excluindo as colunas de clientes pois no momento do modelo em producao nao se tem essa variavel e nao faz sentido colocar a variavel de loja aberta uma vez que ela nao terá um resutado diferente. As colunas de promo_interval e month_map sao excluídas pq foram usadas para derivar colunas

        #seleca de colunas a serem excluídas
        cols_drop = [ 'open','promo_interval', 'month_map']

        #dropando as colunas

        df2 = df2.drop(cols_drop, axis = 1)
   
        return df2
    
    def data_preparation (self, df5):
        
        
    ##Métodos escolhidos para scaling
##Iremos utilizar min-max nas variáveis year e promo_time_week por nao terem outliers
##Iremos utilizar robust scaler nas variáveis competition_time_month e competition_distance por terem outiliers

		# competition_distance
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5[['competition_distance']].values)
	
																			
		# competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5[['competition_time_month']].values)
		
		# promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5[['promo_time_week']].values)
		
		# year 
        df5['year'] = self.year_scaler.transform(df5[['year']].values)
		
		#state_holiday - One Hot Enconodng
		#store_type - Label Enconding
        df5['store_type'] = self.store_type_scaler.transform( df5[['store_type']].values)  

										
		#assortment - Ordinal Enconding
        assortment_dict = {'basic': 1,'extra':2, 'extend':3}
        df5['assortment']=df5['assortment'].map(assortment_dict)
		
		###Natureza ciclica
		# month 
        df5['month_sin']  = df5['month'].apply( lambda x: np.sin( x * (2.  * np.pi/12  ) ) )
        df5['month_cos']  = df5['month'].apply( lambda x: np.cos( x * (2.  * np.pi/12  ) ) )
		
		# day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin ( x *(2 * np.pi/30 ) ) )
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos ( x *(2 * np.pi/30 ) ) )
		
		#week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x : np.sin ( x * (2 * np.pi /52 ) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x : np.cos (x  * (2 * np.pi /52 ) ) )
		
		#day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x : np.sin (x * (2 * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x : np.cos (x * (2 * np.pi/7 ) ) )
		
        cols_selected =      ['store', 'promo','store_type', 'assortment','competition_distance', 'competition_open_since_month','competition_open_since_year','promo2','promo2_since_week', 'promo2_since_year', 'week_of_year','competition_time_month', 'promo_time_week', 'day_sin','day_cos','day_of_week_sin', 'day_of_week_cos']

            
        return df5 [ cols_selected ]

    def get_prediction (self,model,original_data, test_data):
        #prediction
        pred = model.predict (test_data)
                  
        #join pred into the original data
        original_data['prediction'] = np.expm1( pred )
        
                 
        return original_data.to_json ( orient ='records', date_format = 'iso' )