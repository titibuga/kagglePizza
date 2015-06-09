from sklearn.feature_extraction import DictVectorizer
import numpy as np
from datetime import datetime

class BasicFeatureExtractor:
    def __init__(self):
        self.targetKey = "requester_received_pizza"
        

        self.ignoredKeys = [self.targetKey,
                            "request_title",
                            "request_text",
                            "request_text_edit_aware",
                            "request_id",
                            "requester_username",
                            "requester_subreddits_at_request",
                            "requester_user_flair",
        ]

        self.usefulKeys = ['requester_number_of_posts_on_raop_at_request',
                           #'requester_subreddits_at_request',
                           'requester_number_of_comments_at_request',
                          # 'request_title',
                          # 'giver_username_if_known',
                           'requester_days_since_first_post_on_raop_at_request',
                           'requester_account_age_in_days_at_request',
                           'requester_upvotes_minus_downvotes_at_request',
                          # 'requester_username',
                          # 'unix_timestamp_of_request',
                           'requester_upvotes_plus_downvotes_at_request',
                           'unix_timestamp_of_request_utc',
                           'requester_number_of_posts_at_request',
                          # 'request_text_edit_aware',
                          # 'request_id',
                           'requester_number_of_comments_in_raop_at_request',
                           'requester_number_of_subreddits_at_request']

        
    def fit(self, data, y=None):
        return self

    def transform(self, data):
        data2 = []
        for d in data:
            d2 = {}
            for k in d.keys():
                if k in self.usefulKeys:
                    d2[k] = d[k]
                #Text lengt
                d2['req_len'] = self.req_len(d)
               # d2['years_after_2011'] = self.years_after_2011(d)
                
            data2.append(d2);
                
        return data2
        
    def fit_transform(self,data, y=None):
        return self.transform(data)

    ##### feature functions #####

    def req_len(self,sample):
        msg = ''
        if 'request_text_edit_aware' in sample.keys():
            msg += sample['request_text_edit_aware']
        if 'request_title' in sample.keys():
            msg += sample['request_title']
        return len(msg)

    def years_after_2011(self,sample):
        timestamp = sample['unix_timestamp_of_request_utc']
        date = datetime.fromtimestamp(timestamp)
       # return date.year - 2011
        return timestamp



class MessagesExtractor:
    
    def fit(self, data, y=None):
         return self

    def transform(self, data):
        result = []
        kaware = 'request_text_edit_aware'
        kedited = 'post_was_edited'
        koriginal = 'request_text'  #only in training set
        ktitle = 'request_title'
        
        for d in data:
            if koriginal in d:
                if d[kedited]:
                    result.append(d[kaware])
                else:
                    result.append(d[koriginal])
            else:
                if kaware in d:
                    result.append(d[kaware])
                else:
                    result.append[" "]
            if ktitle in d:
                result[-1] += " "+d[ktitle]           
        return result
        
    def fit_transform(self,data, y=None):
        return self.transform(data)
    
    
        

