# Yugen.ai Assignment 

### Problem Statement

In this assignment, you will analyse an open dataset about a marketing campaign of a Portuguese bank in order to design strategies for improving future marketing campaigns. 

### Objective 

The object of this campaign is to pursue customers to subscribe the term deposit. The Marketing campaign was based on phone calls. 

### Goal

The classification goal is to predict if the client will subscribe a term deposit (variable y). Often, more than one contact to the same client was required, in
order to access if the product (bank term deposit) would be (or not) subscribed.

### Attribute Information:

* age -- a numeric field
* job -- a categorical field representing type of job
* education -- a categorical field representing level of education
* default -- a categorical field representing whether there is a credit default or not
* balance -- average balance (in euros)
* housing -- a categorical field representing if a person has housing loan or not
* loan -- a categorical field representing whether a person has personal loan or not
* contact -- a categorical field representing contact communication type
* day -- a numeric field representing last day of the month contacted
* month -- a categorical field representing the last contact month
* duration -- a numeric field representing last contact duration (in seconds)
* campaign -- a numeric field representing the number of contacts performed during this campaign and for this client
* pdays -- a numeric field representing number of days that passed by after the client was last contacted from a previous campaign
* previous -- a numeric field representing number of contacts performed before this campaign and for this client
* poutcome -- a categorical field representing the outcome of the previous marketing campaign
* y -- a categorical field representing whether a client will subscribe the term deposit or not (**target variable**)

### API Structure

The model is trained using a XGBoost Classifier, and to get a prediction from this model , a JSON value of similar structure has to passed to the API. 

```JSON
{"1":{"age":33,
    "job":"technician",
    "marital":"single",
    "education":"secondary",
    "default":"no",
    "balance":24500,
    "housing":"yes",
    "loan":"yes",
    "contact":"cellular",
    "day":13,
    "month":"jun",
    "duration":198,
    "campaign":4,
    "pdays":-1,
    "previous":0,
    "poutcome":"success"
    },
 "2":{"age":23,
    "job":"unknown",
    "marital":"married",
    "education":"tertiary",
    "default":"yes",
    "balance":24500,
    "housing":"yes",
    "loan":"yes",
    "contact":"cellular",
    "day":8,
    "month":"aug",
    "duration":198,
    "campaign":4,
    "pdays":-1,
    "previous":0,
    "poutcome":"success"
    }}
```

The output is then returned for each record in this JSON.

