
# User behavior as a predictor for input accuracy
Johann Miller, University of Maryland Makeability Lab

### Introduction
[Project Sidewalk](http://sidewalk.umiacs.umd.edu) is an online platform for identifying accessibility problems in sidewalks. Users navigate streets using Google Street View, and place labels on issues such as crosswalks without curbramps, uneven pavement, and obstacles blocking the path. In order to ensure some level of accuracy in the data, Project Sidewalk can use a couple tools. Each user has to complete a tutorial before they can begin to report problems. Another option is ground truth seeding, where users place labels in a region that already has established answers. If the user enters data that doesn't match the ground truth, then all of the data they entered can be flagged.

Here, we will investigate another possible option: using the user's interactions with the tool to predict their accuracy. Consider a user who is inactive for long periods, and barely uses any of the tool's features. This user probably gives worse input than a user who works consistently and employs all of the tool's features. Aspects of interaction include mouse movement, keypresses, and others that we can collect while users place labels. If accurate labels correspond to a certain type of usage, then these features could predict the accuracy of a user even in non ground truth regions.

To see if this is possible, we'll use data from ground truth regions to train and test a classifier.

### Setup python notebook


```python
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn import multiclass
from sklearn import svm

```


```python
%%html
<style>
img {margin-left: 0!important} /* left align images */
</style>
```


<style>
img {margin-left: 0!important} /* left align images */
</style>


### Collecting user events
Project Sidewalk has logs for variety of user events. The events range from low-level (mouse movements, clicks, etc.) to high-level (zoom in/out, changing label mode, etc.). If we query the interaction table, we can see all the event types.


```python
event_types = pd.read_csv('data/interim/event-types.csv')
print('Num of event types:', len(event_types))
event_types.head()
```

    Num of event types: 109





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Click_LabelDelete</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Click_ModeSwitch_CurbRamp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Click_ModeSwitch_NoCurbRamp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Click_ModeSwitch_NoSidewalk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Click_ModeSwitch_Obstacle</td>
    </tr>
  </tbody>
</table>
</div>



For each user session, we have a collection of the events that were triggered. In order to compare two sessions, we can look at the total number of each type of event. We also look at the mean and standard deviation of the number of events per Google Street View panorama. This way, a user session in a large region can be compared fairly to a session in a smaller region since the former will have more panoramas.

We can load in these event counts from `features.csv`. This file was created by **TODO**.


```python
features = pd.read_csv('data/interim/feature-names.csv')
print('Num of features:', len(features))
features.head()
```

    Num of features: 327





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Click_LabelDelete_per_pan_mean</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Click_LabelDelete_per_pan_std</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Click_LabelDelete_total</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Click_ModeSwitch_CurbRamp_per_pan_mean</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Click_ModeSwitch_CurbRamp_per_pan_std</td>
    </tr>
  </tbody>
</table>
</div>



### Grading user accuracy
We need to rate each user session on its accuracy compared to the ground truth. We do so by counting the number of true positives, false positives, true negatives, and false negatives from the session. These are defined as follows:

#### True positive
The user placed a correct label. Here, the green icon is a label for a curb ramp. The user placed it correctly, so this is a true positive.

![true positive](images/true-pos.png)

#### False positive
The user placed an incorrect label. Here, the user placed a green icon to identify a curb ramp, but none are present.

![false positive](images/false-pos.png)

#### True negative
There was nothing to label, and the user didn't label anything. The amount of empty space necessary to be considered a true negative is the `granularity`. The scoring was done with three values: 5 meter, 10 meter, and street.

![true negative](images/true-neg.png)

#### False negative
There was something to label, but the user missed it. Here, there is a curb ramp with no label.

![false negative](images/false-neg.png)


```python
raw_scores = pd.read_csv('data/interim/processing/scores.csv')
raw_scores.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition_id</th>
      <th>worker_id</th>
      <th>type</th>
      <th>granularity</th>
      <th>label.type</th>
      <th>true_pos</th>
      <th>false_pos</th>
      <th>true_neg</th>
      <th>false_neg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70</td>
      <td>01232fef-5a19-4435-8be6-c0da3b38cabd</td>
      <td>volunteer</td>
      <td>5_meter</td>
      <td>Problem</td>
      <td>27</td>
      <td>26</td>
      <td>186</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>70</td>
      <td>01232fef-5a19-4435-8be6-c0da3b38cabd</td>
      <td>volunteer</td>
      <td>10_meter</td>
      <td>Problem</td>
      <td>27</td>
      <td>15</td>
      <td>76</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>70</td>
      <td>01232fef-5a19-4435-8be6-c0da3b38cabd</td>
      <td>volunteer</td>
      <td>street</td>
      <td>Problem</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>72</td>
      <td>9501513f-3822-4921-861e-8f1440dee102</td>
      <td>volunteer</td>
      <td>5_meter</td>
      <td>Problem</td>
      <td>22</td>
      <td>58</td>
      <td>161</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>72</td>
      <td>9501513f-3822-4921-861e-8f1440dee102</td>
      <td>volunteer</td>
      <td>10_meter</td>
      <td>Problem</td>
      <td>21</td>
      <td>44</td>
      <td>59</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Composite scores
We can combine the counts of true positives, true negatives, etc. into metrics that tell us about the accuracy of the user.

$\text{accuracy} = \frac{\text{true positives + true negatives}}{\text{true positives + false positives + true negatives + false negatives}}$

$\text{positive predictive value (ppv)} = \frac{\text{true positives}}{\text{true positives + false positives}}$

$\text{negative predictive value (npv)} = \frac{\text{true negatives}}{\text{true negatives + false negatives}}$

$\text{recall} = \frac{\text{true positives}}{\text{true positives + false negatives}}$

$\text{specificity} = \frac{\text{true positives}}{\text{true positives + true negatives}}$


```python
scores = pd.read_csv('data/interim/processing/scores-comb-acc.csv')
scores[scores.columns[4:]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true_pos</th>
      <th>false_pos</th>
      <th>true_neg</th>
      <th>false_neg</th>
      <th>accuracy</th>
      <th>ppv</th>
      <th>npv</th>
      <th>recall</th>
      <th>specificity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>116</td>
      <td>86</td>
      <td>1564</td>
      <td>54</td>
      <td>0.923077</td>
      <td>0.574257</td>
      <td>0.966625</td>
      <td>0.682353</td>
      <td>0.947879</td>
    </tr>
    <tr>
      <th>1</th>
      <td>112</td>
      <td>136</td>
      <td>3262</td>
      <td>74</td>
      <td>0.941406</td>
      <td>0.451613</td>
      <td>0.977818</td>
      <td>0.602151</td>
      <td>0.959976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76</td>
      <td>18</td>
      <td>140</td>
      <td>4</td>
      <td>0.907563</td>
      <td>0.808511</td>
      <td>0.972222</td>
      <td>0.950000</td>
      <td>0.886076</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120</td>
      <td>110</td>
      <td>1564</td>
      <td>99</td>
      <td>0.889593</td>
      <td>0.521739</td>
      <td>0.940469</td>
      <td>0.547945</td>
      <td>0.934289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95</td>
      <td>151</td>
      <td>3264</td>
      <td>132</td>
      <td>0.922295</td>
      <td>0.386179</td>
      <td>0.961131</td>
      <td>0.418502</td>
      <td>0.955783</td>
    </tr>
  </tbody>
</table>
</div>



### Data exploration
Now that we've collected features for user interaction and labeled them based on accuracy, we can begin to examine it.


```python
from sklearn import preprocessing

data = pd.read_csv('data/final/interaction-comb.csv')

label_cols = data.columns[8:13].tolist()
feature_cols = data.columns[13:].tolist()
```

#### Shape of the data


```python
print('Num of users (samples):', len(data[['condition_id', 'worker_id']].drop_duplicates()))
print('  Num of turkers:', len(data[data.type == 'turker'][['condition_id', 'worker_id']].drop_duplicates()))
print('  Num of volunteers:', len(data[data.type == 'volunteer'][['condition_id', 'worker_id']].drop_duplicates()))
print('Num of features:', len(feature_cols))
```

    Num of users (samples): 308
      Num of turkers: 264
      Num of volunteers: 44
    Num of features: 291


We almost have more features than samples, meaning our data may be sparse. We can look into the complexity of the data by finding the correlations of the features.

#### Feature correlation


```python
# get correlation matrix
features = data[data.granularity == 'street'][feature_cols]
features = pd.DataFrame(preprocessing.scale(features))
correlations = features.corr().fillna(0)

# plot matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()

# print average correlation
c = correlations.unstack()
c = (c[c.index.get_level_values(0) > c.index.get_level_values(1)]).sort_values() # select above diagonal
print('Average absolute feature correlation:', c.abs().mean(), '\n')
```


![png](images/output_17_0.png)


    Average absolute feature correlation: 0.116891488326 
    


We do have correlations between many of the features. Next we can look at the specific features with the most correlation.


```python
col1,col2 = c.index.get_level_values(0), c.index.get_level_values(1)
print('Highest correlations:')
for i in range(len(c) - 3, len(c)):
        print((feature_cols[col1[i]], feature_cols[col2[i]]), c.iloc[i])      
print('\nInverse correlations:')
for i in range(0, 3):
    print((feature_cols[col1[i]], feature_cols[col2[i]]), c.iloc[i])
c = c.abs().sort_values()
col1,col2 = c.index.get_level_values(0), c.index.get_level_values(1)
print('\nLowest correlations:')
for i in range(0, 3):
    print((feature_cols[col1[i]], feature_cols[col2[i]]), c.iloc[i])
```

    Highest correlations:
    ('PopUpMessage_SignUpClickYes_total', 'PopUpMessage_SignUpClickYes_per_pan_mean') 1.0
    ('TaskSkip_per_pan_mean', 'ModalSkip_ClickRadio_per_pan_mean') 1.0
    ('ModalSkip_ClickRadio_total', 'ModalSkip_ClickOK_total') 1.0
    
    Inverse correlations:
    ('TaskStart_total', 'PopUpShow_GSVLabelDisappear_per_pan_mean') -0.633997145207
    ('TaskEnd_total', 'PopUpShow_GSVLabelDisappear_per_pan_mean') -0.60885617748
    ('TaskStart_total', 'PopUpShow_LetsGetStarted_per_pan_mean') -0.570528652
    
    Lowest correlations:
    ('Unload_total', 'KeyboardShortcut_ModeSwitch_Occlusion_total') 1.49591738708e-17
    ('POV_Changed_total', 'Click_Undo_per_pan_std') 4.16556249927e-06
    ('TaskSubmit_per_pan_mean', 'Click_ModeSwitch_Obstacle_total') 1.29480088958e-05


Looking at the highest correlations reveals that we have reduntant features. `TaskSkip`, `ModalSkip_ClickRadio`, and `ModalSkip_ClickOK` are all triggered during the same sequence of events. `PopUpMessage_SignUpClickYes` is an event that didn't occur often, and in every occurence the user only visited one panorama. Hence the total and per panorama counts are identicical.

The inverse correlations are between events that occur at regular times. Tasks start and end once per panorama. The popups occur only once at the beginning of a session. As the number of panoramas in the session increases, the total number of `TaskStart` events increases and the number of `PopUpShow` events decrease. As a sanity check, we can view the features with the least correlation. These features are for unrelated events, which is what we would expect.

The presence of reduntant and negatively correlated features is a sign that the data may be less complex than the number of dimensions suggest.

#### Label distribution
As previously discussed, we have several composite scores that we can use to describe the quality of a user's input. We also have 3 levels of granularity for the scores. Here we can look at the distributions for each.


```python
gran_types = ['5_meter', '10_meter', 'street']
fig, ax = plt.subplots(ncols=3, sharey=True)
for i, gran in enumerate(gran_types):
    labels = data[data.granularity == gran][label_cols] # labels with given granularity
    s = labels.stack().reset_index() # condense all score types to one column
    s.columns = ['index', 'score_type', 'score']
    
    b = sns.boxplot(x=s['score_type'], y=s['score'], ax=ax[i])
    b.set_title(gran + ' granularity')   
fig.set_size_inches(12, 3.5)
fig.tight_layout()
```


![png](images/output_21_0.png)


Except for precision (ppv) and recall, the scores are much higher and with less variation for 5 and 10 meter granularity. In order to understand why, we can look at the raw counts of true positives, false posivies, etc.


```python
raw_score_cols = ['true_pos', 'false_pos', 'true_neg', 'false_neg']
fig, ax = plt.subplots(ncols=3, sharey=True)
for i, gran in enumerate(gran_types):
    labels = data[data.granularity == gran][raw_score_cols] # tp,fp,tn,fn for given granularity
    s = labels.stack().reset_index() # move all score types to one column
    s.columns = ['index', 'score', 'count']
    
    ax[i].set_yscale('log')
    b = sns.boxplot(x=s['score'], y=s['count'], ax=ax[i])
    b.set_title(gran + ' granularity')
    
fig.set_size_inches(12, 3.5)
fig.tight_layout()
```


![png](images/output_23_0.png)


5 and 10 meter granularity have an order of magnitude more true negatives than other the other scores. The smaller granularity means that less area is required to be considered a true negative, and thus many more true negatives occur. Precision and recall were not affected much because they do not depend on the number of true negatives. The other three scores were inflated by the disparate number of true negatives. To avoid this effect, we will use street level granularity to judge users.

#### Label correlation
To further understand our labels for the data, we can look at the relationship between the different scores.


```python
# draw correlation scatterplots
labels = data[data.granularity == 'street'][label_cols]
scatter_matrix(labels, diagonal='hist')

# draw correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(labels.corr(), vmin=-1, vmax=1)
ax.set_xticklabels([''] + label_cols)
ax.set_yticklabels([''] + label_cols)
fig.colorbar(cax)
plt.show()
```


![png](images/output_25_0.png)



![png](images/output_25_1.png)


Scores like precision (ppv) and recall are inversely correlated. Accuracy has a positive correlation with every score. It also takes data into account, so we will use it as the main judge of user performance.

#### Principal Component Analysis
Now that we have looked at the features and lables, we can look for patterns between them. Principal component analysis combines features to reduce dimensionality of the data with a minimal loss of information.


```python
# Create a regular PCA model 
pca = decomposition.PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(features)


colormap = branca.colormap.linear.RdYlGn.scale(labels.accuracy.min().round(1), labels.accuracy.max().round(1))
step = 0.1
for i in np.arange(0, 1, step): # for each bucket
    i = 1 - i - step # reverse
    x = reduced_data_pca[:,0][(labels.accuracy >= i) & (labels.accuracy < i + step)] # first comp
    y = reduced_data_pca[:,1][(labels.accuracy >= i) & (labels.accuracy < i + step)] # second comp
    plt.scatter(x=x, y=y, color=colormap(i + step/2), alpha=0.8)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.title("PCA")
plt.show()
print('User Accuracy')
colormap
```


![png](images/output_27_0.png)


    User Accuracy





<svg height="50" width="500"><line x1="0" y1="0" x2="0" y2="20" style="stroke:#d73027;stroke-width:3;" /><line x1="1" y1="0" x2="1" y2="20" style="stroke:#d83127;stroke-width:3;" /><line x1="2" y1="0" x2="2" y2="20" style="stroke:#d83228;stroke-width:3;" /><line x1="3" y1="0" x2="3" y2="20" style="stroke:#d83228;stroke-width:3;" /><line x1="4" y1="0" x2="4" y2="20" style="stroke:#d93329;stroke-width:3;" /><line x1="5" y1="0" x2="5" y2="20" style="stroke:#d93429;stroke-width:3;" /><line x1="6" y1="0" x2="6" y2="20" style="stroke:#da352a;stroke-width:3;" /><line x1="7" y1="0" x2="7" y2="20" style="stroke:#da362a;stroke-width:3;" /><line x1="8" y1="0" x2="8" y2="20" style="stroke:#da372b;stroke-width:3;" /><line x1="9" y1="0" x2="9" y2="20" style="stroke:#db382b;stroke-width:3;" /><line x1="10" y1="0" x2="10" y2="20" style="stroke:#db392c;stroke-width:3;" /><line x1="11" y1="0" x2="11" y2="20" style="stroke:#db3a2c;stroke-width:3;" /><line x1="12" y1="0" x2="12" y2="20" style="stroke:#dc3b2d;stroke-width:3;" /><line x1="13" y1="0" x2="13" y2="20" style="stroke:#dc3c2d;stroke-width:3;" /><line x1="14" y1="0" x2="14" y2="20" style="stroke:#dd3d2e;stroke-width:3;" /><line x1="15" y1="0" x2="15" y2="20" style="stroke:#dd3e2e;stroke-width:3;" /><line x1="16" y1="0" x2="16" y2="20" style="stroke:#dd3f2f;stroke-width:3;" /><line x1="17" y1="0" x2="17" y2="20" style="stroke:#de402f;stroke-width:3;" /><line x1="18" y1="0" x2="18" y2="20" style="stroke:#de4130;stroke-width:3;" /><line x1="19" y1="0" x2="19" y2="20" style="stroke:#de4130;stroke-width:3;" /><line x1="20" y1="0" x2="20" y2="20" style="stroke:#df4231;stroke-width:3;" /><line x1="21" y1="0" x2="21" y2="20" style="stroke:#df4331;stroke-width:3;" /><line x1="22" y1="0" x2="22" y2="20" style="stroke:#e04432;stroke-width:3;" /><line x1="23" y1="0" x2="23" y2="20" style="stroke:#e04532;stroke-width:3;" /><line x1="24" y1="0" x2="24" y2="20" style="stroke:#e04633;stroke-width:3;" /><line x1="25" y1="0" x2="25" y2="20" style="stroke:#e14733;stroke-width:3;" /><line x1="26" y1="0" x2="26" y2="20" style="stroke:#e14834;stroke-width:3;" /><line x1="27" y1="0" x2="27" y2="20" style="stroke:#e14934;stroke-width:3;" /><line x1="28" y1="0" x2="28" y2="20" style="stroke:#e24a35;stroke-width:3;" /><line x1="29" y1="0" x2="29" y2="20" style="stroke:#e24b35;stroke-width:3;" /><line x1="30" y1="0" x2="30" y2="20" style="stroke:#e34c36;stroke-width:3;" /><line x1="31" y1="0" x2="31" y2="20" style="stroke:#e34d36;stroke-width:3;" /><line x1="32" y1="0" x2="32" y2="20" style="stroke:#e34e37;stroke-width:3;" /><line x1="33" y1="0" x2="33" y2="20" style="stroke:#e44f37;stroke-width:3;" /><line x1="34" y1="0" x2="34" y2="20" style="stroke:#e44f38;stroke-width:3;" /><line x1="35" y1="0" x2="35" y2="20" style="stroke:#e45038;stroke-width:3;" /><line x1="36" y1="0" x2="36" y2="20" style="stroke:#e55139;stroke-width:3;" /><line x1="37" y1="0" x2="37" y2="20" style="stroke:#e55239;stroke-width:3;" /><line x1="38" y1="0" x2="38" y2="20" style="stroke:#e5533a;stroke-width:3;" /><line x1="39" y1="0" x2="39" y2="20" style="stroke:#e6543a;stroke-width:3;" /><line x1="40" y1="0" x2="40" y2="20" style="stroke:#e6553b;stroke-width:3;" /><line x1="41" y1="0" x2="41" y2="20" style="stroke:#e7563b;stroke-width:3;" /><line x1="42" y1="0" x2="42" y2="20" style="stroke:#e7573c;stroke-width:3;" /><line x1="43" y1="0" x2="43" y2="20" style="stroke:#e7583c;stroke-width:3;" /><line x1="44" y1="0" x2="44" y2="20" style="stroke:#e8593d;stroke-width:3;" /><line x1="45" y1="0" x2="45" y2="20" style="stroke:#e85a3d;stroke-width:3;" /><line x1="46" y1="0" x2="46" y2="20" style="stroke:#e85b3e;stroke-width:3;" /><line x1="47" y1="0" x2="47" y2="20" style="stroke:#e95c3e;stroke-width:3;" /><line x1="48" y1="0" x2="48" y2="20" style="stroke:#e95d3f;stroke-width:3;" /><line x1="49" y1="0" x2="49" y2="20" style="stroke:#ea5e3f;stroke-width:3;" /><line x1="50" y1="0" x2="50" y2="20" style="stroke:#ea5e40;stroke-width:3;" /><line x1="51" y1="0" x2="51" y2="20" style="stroke:#ea5f40;stroke-width:3;" /><line x1="52" y1="0" x2="52" y2="20" style="stroke:#eb6041;stroke-width:3;" /><line x1="53" y1="0" x2="53" y2="20" style="stroke:#eb6141;stroke-width:3;" /><line x1="54" y1="0" x2="54" y2="20" style="stroke:#eb6242;stroke-width:3;" /><line x1="55" y1="0" x2="55" y2="20" style="stroke:#ec6342;stroke-width:3;" /><line x1="56" y1="0" x2="56" y2="20" style="stroke:#ec6443;stroke-width:3;" /><line x1="57" y1="0" x2="57" y2="20" style="stroke:#ed6543;stroke-width:3;" /><line x1="58" y1="0" x2="58" y2="20" style="stroke:#ed6644;stroke-width:3;" /><line x1="59" y1="0" x2="59" y2="20" style="stroke:#ed6744;stroke-width:3;" /><line x1="60" y1="0" x2="60" y2="20" style="stroke:#ee6845;stroke-width:3;" /><line x1="61" y1="0" x2="61" y2="20" style="stroke:#ee6945;stroke-width:3;" /><line x1="62" y1="0" x2="62" y2="20" style="stroke:#ee6a46;stroke-width:3;" /><line x1="63" y1="0" x2="63" y2="20" style="stroke:#ef6b46;stroke-width:3;" /><line x1="64" y1="0" x2="64" y2="20" style="stroke:#ef6c47;stroke-width:3;" /><line x1="65" y1="0" x2="65" y2="20" style="stroke:#f06c47;stroke-width:3;" /><line x1="66" y1="0" x2="66" y2="20" style="stroke:#f06d48;stroke-width:3;" /><line x1="67" y1="0" x2="67" y2="20" style="stroke:#f06e48;stroke-width:3;" /><line x1="68" y1="0" x2="68" y2="20" style="stroke:#f16f49;stroke-width:3;" /><line x1="69" y1="0" x2="69" y2="20" style="stroke:#f17049;stroke-width:3;" /><line x1="70" y1="0" x2="70" y2="20" style="stroke:#f1714a;stroke-width:3;" /><line x1="71" y1="0" x2="71" y2="20" style="stroke:#f2724a;stroke-width:3;" /><line x1="72" y1="0" x2="72" y2="20" style="stroke:#f2734b;stroke-width:3;" /><line x1="73" y1="0" x2="73" y2="20" style="stroke:#f3744b;stroke-width:3;" /><line x1="74" y1="0" x2="74" y2="20" style="stroke:#f3754c;stroke-width:3;" /><line x1="75" y1="0" x2="75" y2="20" style="stroke:#f3764c;stroke-width:3;" /><line x1="76" y1="0" x2="76" y2="20" style="stroke:#f4774d;stroke-width:3;" /><line x1="77" y1="0" x2="77" y2="20" style="stroke:#f4784d;stroke-width:3;" /><line x1="78" y1="0" x2="78" y2="20" style="stroke:#f4794e;stroke-width:3;" /><line x1="79" y1="0" x2="79" y2="20" style="stroke:#f57a4e;stroke-width:3;" /><line x1="80" y1="0" x2="80" y2="20" style="stroke:#f57b4f;stroke-width:3;" /><line x1="81" y1="0" x2="81" y2="20" style="stroke:#f57b4f;stroke-width:3;" /><line x1="82" y1="0" x2="82" y2="20" style="stroke:#f67c50;stroke-width:3;" /><line x1="83" y1="0" x2="83" y2="20" style="stroke:#f67d50;stroke-width:3;" /><line x1="84" y1="0" x2="84" y2="20" style="stroke:#f77e51;stroke-width:3;" /><line x1="85" y1="0" x2="85" y2="20" style="stroke:#f77f51;stroke-width:3;" /><line x1="86" y1="0" x2="86" y2="20" style="stroke:#f78052;stroke-width:3;" /><line x1="87" y1="0" x2="87" y2="20" style="stroke:#f88152;stroke-width:3;" /><line x1="88" y1="0" x2="88" y2="20" style="stroke:#f88253;stroke-width:3;" /><line x1="89" y1="0" x2="89" y2="20" style="stroke:#f88353;stroke-width:3;" /><line x1="90" y1="0" x2="90" y2="20" style="stroke:#f98454;stroke-width:3;" /><line x1="91" y1="0" x2="91" y2="20" style="stroke:#f98554;stroke-width:3;" /><line x1="92" y1="0" x2="92" y2="20" style="stroke:#fa8655;stroke-width:3;" /><line x1="93" y1="0" x2="93" y2="20" style="stroke:#fa8755;stroke-width:3;" /><line x1="94" y1="0" x2="94" y2="20" style="stroke:#fa8856;stroke-width:3;" /><line x1="95" y1="0" x2="95" y2="20" style="stroke:#fb8956;stroke-width:3;" /><line x1="96" y1="0" x2="96" y2="20" style="stroke:#fb8957;stroke-width:3;" /><line x1="97" y1="0" x2="97" y2="20" style="stroke:#fb8a57;stroke-width:3;" /><line x1="98" y1="0" x2="98" y2="20" style="stroke:#fc8b58;stroke-width:3;" /><line x1="99" y1="0" x2="99" y2="20" style="stroke:#fc8c58;stroke-width:3;" /><line x1="100" y1="0" x2="100" y2="20" style="stroke:#fc8d59;stroke-width:3;" /><line x1="101" y1="0" x2="101" y2="20" style="stroke:#fd8e59;stroke-width:3;" /><line x1="102" y1="0" x2="102" y2="20" style="stroke:#fd8f5a;stroke-width:3;" /><line x1="103" y1="0" x2="103" y2="20" style="stroke:#fd905a;stroke-width:3;" /><line x1="104" y1="0" x2="104" y2="20" style="stroke:#fd915b;stroke-width:3;" /><line x1="105" y1="0" x2="105" y2="20" style="stroke:#fd915b;stroke-width:3;" /><line x1="106" y1="0" x2="106" y2="20" style="stroke:#fd925c;stroke-width:3;" /><line x1="107" y1="0" x2="107" y2="20" style="stroke:#fd935c;stroke-width:3;" /><line x1="108" y1="0" x2="108" y2="20" style="stroke:#fd945d;stroke-width:3;" /><line x1="109" y1="0" x2="109" y2="20" style="stroke:#fd955d;stroke-width:3;" /><line x1="110" y1="0" x2="110" y2="20" style="stroke:#fd965e;stroke-width:3;" /><line x1="111" y1="0" x2="111" y2="20" style="stroke:#fd965e;stroke-width:3;" /><line x1="112" y1="0" x2="112" y2="20" style="stroke:#fd975f;stroke-width:3;" /><line x1="113" y1="0" x2="113" y2="20" style="stroke:#fd985f;stroke-width:3;" /><line x1="114" y1="0" x2="114" y2="20" style="stroke:#fd9960;stroke-width:3;" /><line x1="115" y1="0" x2="115" y2="20" style="stroke:#fd9a60;stroke-width:3;" /><line x1="116" y1="0" x2="116" y2="20" style="stroke:#fd9b61;stroke-width:3;" /><line x1="117" y1="0" x2="117" y2="20" style="stroke:#fd9b62;stroke-width:3;" /><line x1="118" y1="0" x2="118" y2="20" style="stroke:#fd9c62;stroke-width:3;" /><line x1="119" y1="0" x2="119" y2="20" style="stroke:#fd9d63;stroke-width:3;" /><line x1="120" y1="0" x2="120" y2="20" style="stroke:#fd9e63;stroke-width:3;" /><line x1="121" y1="0" x2="121" y2="20" style="stroke:#fd9f64;stroke-width:3;" /><line x1="122" y1="0" x2="122" y2="20" style="stroke:#fda064;stroke-width:3;" /><line x1="123" y1="0" x2="123" y2="20" style="stroke:#fda065;stroke-width:3;" /><line x1="124" y1="0" x2="124" y2="20" style="stroke:#fda165;stroke-width:3;" /><line x1="125" y1="0" x2="125" y2="20" style="stroke:#fda266;stroke-width:3;" /><line x1="126" y1="0" x2="126" y2="20" style="stroke:#fda366;stroke-width:3;" /><line x1="127" y1="0" x2="127" y2="20" style="stroke:#fda467;stroke-width:3;" /><line x1="128" y1="0" x2="128" y2="20" style="stroke:#fda567;stroke-width:3;" /><line x1="129" y1="0" x2="129" y2="20" style="stroke:#fda568;stroke-width:3;" /><line x1="130" y1="0" x2="130" y2="20" style="stroke:#fda668;stroke-width:3;" /><line x1="131" y1="0" x2="131" y2="20" style="stroke:#fda769;stroke-width:3;" /><line x1="132" y1="0" x2="132" y2="20" style="stroke:#fda869;stroke-width:3;" /><line x1="133" y1="0" x2="133" y2="20" style="stroke:#fda96a;stroke-width:3;" /><line x1="134" y1="0" x2="134" y2="20" style="stroke:#fdaa6a;stroke-width:3;" /><line x1="135" y1="0" x2="135" y2="20" style="stroke:#fdaa6b;stroke-width:3;" /><line x1="136" y1="0" x2="136" y2="20" style="stroke:#fdab6b;stroke-width:3;" /><line x1="137" y1="0" x2="137" y2="20" style="stroke:#fdac6c;stroke-width:3;" /><line x1="138" y1="0" x2="138" y2="20" style="stroke:#fdad6c;stroke-width:3;" /><line x1="139" y1="0" x2="139" y2="20" style="stroke:#fdae6d;stroke-width:3;" /><line x1="140" y1="0" x2="140" y2="20" style="stroke:#fdaf6d;stroke-width:3;" /><line x1="141" y1="0" x2="141" y2="20" style="stroke:#fdaf6e;stroke-width:3;" /><line x1="142" y1="0" x2="142" y2="20" style="stroke:#fdb06e;stroke-width:3;" /><line x1="143" y1="0" x2="143" y2="20" style="stroke:#fdb16f;stroke-width:3;" /><line x1="144" y1="0" x2="144" y2="20" style="stroke:#fdb26f;stroke-width:3;" /><line x1="145" y1="0" x2="145" y2="20" style="stroke:#fdb370;stroke-width:3;" /><line x1="146" y1="0" x2="146" y2="20" style="stroke:#fdb470;stroke-width:3;" /><line x1="147" y1="0" x2="147" y2="20" style="stroke:#fdb471;stroke-width:3;" /><line x1="148" y1="0" x2="148" y2="20" style="stroke:#fdb571;stroke-width:3;" /><line x1="149" y1="0" x2="149" y2="20" style="stroke:#fdb672;stroke-width:3;" /><line x1="150" y1="0" x2="150" y2="20" style="stroke:#fdb772;stroke-width:3;" /><line x1="151" y1="0" x2="151" y2="20" style="stroke:#feb873;stroke-width:3;" /><line x1="152" y1="0" x2="152" y2="20" style="stroke:#feb973;stroke-width:3;" /><line x1="153" y1="0" x2="153" y2="20" style="stroke:#feb974;stroke-width:3;" /><line x1="154" y1="0" x2="154" y2="20" style="stroke:#feba74;stroke-width:3;" /><line x1="155" y1="0" x2="155" y2="20" style="stroke:#febb75;stroke-width:3;" /><line x1="156" y1="0" x2="156" y2="20" style="stroke:#febc75;stroke-width:3;" /><line x1="157" y1="0" x2="157" y2="20" style="stroke:#febd76;stroke-width:3;" /><line x1="158" y1="0" x2="158" y2="20" style="stroke:#febe76;stroke-width:3;" /><line x1="159" y1="0" x2="159" y2="20" style="stroke:#febe77;stroke-width:3;" /><line x1="160" y1="0" x2="160" y2="20" style="stroke:#febf77;stroke-width:3;" /><line x1="161" y1="0" x2="161" y2="20" style="stroke:#fec078;stroke-width:3;" /><line x1="162" y1="0" x2="162" y2="20" style="stroke:#fec178;stroke-width:3;" /><line x1="163" y1="0" x2="163" y2="20" style="stroke:#fec279;stroke-width:3;" /><line x1="164" y1="0" x2="164" y2="20" style="stroke:#fec379;stroke-width:3;" /><line x1="165" y1="0" x2="165" y2="20" style="stroke:#fec37a;stroke-width:3;" /><line x1="166" y1="0" x2="166" y2="20" style="stroke:#fec47a;stroke-width:3;" /><line x1="167" y1="0" x2="167" y2="20" style="stroke:#fec57b;stroke-width:3;" /><line x1="168" y1="0" x2="168" y2="20" style="stroke:#fec67b;stroke-width:3;" /><line x1="169" y1="0" x2="169" y2="20" style="stroke:#fec77c;stroke-width:3;" /><line x1="170" y1="0" x2="170" y2="20" style="stroke:#fec87c;stroke-width:3;" /><line x1="171" y1="0" x2="171" y2="20" style="stroke:#fec87d;stroke-width:3;" /><line x1="172" y1="0" x2="172" y2="20" style="stroke:#fec97d;stroke-width:3;" /><line x1="173" y1="0" x2="173" y2="20" style="stroke:#feca7e;stroke-width:3;" /><line x1="174" y1="0" x2="174" y2="20" style="stroke:#fecb7e;stroke-width:3;" /><line x1="175" y1="0" x2="175" y2="20" style="stroke:#fecc7f;stroke-width:3;" /><line x1="176" y1="0" x2="176" y2="20" style="stroke:#fecd7f;stroke-width:3;" /><line x1="177" y1="0" x2="177" y2="20" style="stroke:#fece80;stroke-width:3;" /><line x1="178" y1="0" x2="178" y2="20" style="stroke:#fece80;stroke-width:3;" /><line x1="179" y1="0" x2="179" y2="20" style="stroke:#fecf81;stroke-width:3;" /><line x1="180" y1="0" x2="180" y2="20" style="stroke:#fed081;stroke-width:3;" /><line x1="181" y1="0" x2="181" y2="20" style="stroke:#fed182;stroke-width:3;" /><line x1="182" y1="0" x2="182" y2="20" style="stroke:#fed282;stroke-width:3;" /><line x1="183" y1="0" x2="183" y2="20" style="stroke:#fed383;stroke-width:3;" /><line x1="184" y1="0" x2="184" y2="20" style="stroke:#fed383;stroke-width:3;" /><line x1="185" y1="0" x2="185" y2="20" style="stroke:#fed484;stroke-width:3;" /><line x1="186" y1="0" x2="186" y2="20" style="stroke:#fed584;stroke-width:3;" /><line x1="187" y1="0" x2="187" y2="20" style="stroke:#fed685;stroke-width:3;" /><line x1="188" y1="0" x2="188" y2="20" style="stroke:#fed785;stroke-width:3;" /><line x1="189" y1="0" x2="189" y2="20" style="stroke:#fed886;stroke-width:3;" /><line x1="190" y1="0" x2="190" y2="20" style="stroke:#fed886;stroke-width:3;" /><line x1="191" y1="0" x2="191" y2="20" style="stroke:#fed987;stroke-width:3;" /><line x1="192" y1="0" x2="192" y2="20" style="stroke:#feda87;stroke-width:3;" /><line x1="193" y1="0" x2="193" y2="20" style="stroke:#fedb88;stroke-width:3;" /><line x1="194" y1="0" x2="194" y2="20" style="stroke:#fedc88;stroke-width:3;" /><line x1="195" y1="0" x2="195" y2="20" style="stroke:#fedd89;stroke-width:3;" /><line x1="196" y1="0" x2="196" y2="20" style="stroke:#fedd89;stroke-width:3;" /><line x1="197" y1="0" x2="197" y2="20" style="stroke:#fede8a;stroke-width:3;" /><line x1="198" y1="0" x2="198" y2="20" style="stroke:#fedf8a;stroke-width:3;" /><line x1="199" y1="0" x2="199" y2="20" style="stroke:#fee08b;stroke-width:3;" /><line x1="200" y1="0" x2="200" y2="20" style="stroke:#fee08b;stroke-width:3;" /><line x1="201" y1="0" x2="201" y2="20" style="stroke:#fee18b;stroke-width:3;" /><line x1="202" y1="0" x2="202" y2="20" style="stroke:#fee18b;stroke-width:3;" /><line x1="203" y1="0" x2="203" y2="20" style="stroke:#fde18b;stroke-width:3;" /><line x1="204" y1="0" x2="204" y2="20" style="stroke:#fde18b;stroke-width:3;" /><line x1="205" y1="0" x2="205" y2="20" style="stroke:#fce18b;stroke-width:3;" /><line x1="206" y1="0" x2="206" y2="20" style="stroke:#fce18b;stroke-width:3;" /><line x1="207" y1="0" x2="207" y2="20" style="stroke:#fce18b;stroke-width:3;" /><line x1="208" y1="0" x2="208" y2="20" style="stroke:#fbe28b;stroke-width:3;" /><line x1="209" y1="0" x2="209" y2="20" style="stroke:#fbe28b;stroke-width:3;" /><line x1="210" y1="0" x2="210" y2="20" style="stroke:#fbe28b;stroke-width:3;" /><line x1="211" y1="0" x2="211" y2="20" style="stroke:#fae28b;stroke-width:3;" /><line x1="212" y1="0" x2="212" y2="20" style="stroke:#fae28b;stroke-width:3;" /><line x1="213" y1="0" x2="213" y2="20" style="stroke:#fae28b;stroke-width:3;" /><line x1="214" y1="0" x2="214" y2="20" style="stroke:#f9e38b;stroke-width:3;" /><line x1="215" y1="0" x2="215" y2="20" style="stroke:#f9e38b;stroke-width:3;" /><line x1="216" y1="0" x2="216" y2="20" style="stroke:#f8e38b;stroke-width:3;" /><line x1="217" y1="0" x2="217" y2="20" style="stroke:#f8e38b;stroke-width:3;" /><line x1="218" y1="0" x2="218" y2="20" style="stroke:#f8e38b;stroke-width:3;" /><line x1="219" y1="0" x2="219" y2="20" style="stroke:#f7e38b;stroke-width:3;" /><line x1="220" y1="0" x2="220" y2="20" style="stroke:#f7e38b;stroke-width:3;" /><line x1="221" y1="0" x2="221" y2="20" style="stroke:#f7e48b;stroke-width:3;" /><line x1="222" y1="0" x2="222" y2="20" style="stroke:#f6e48b;stroke-width:3;" /><line x1="223" y1="0" x2="223" y2="20" style="stroke:#f6e48b;stroke-width:3;" /><line x1="224" y1="0" x2="224" y2="20" style="stroke:#f5e48b;stroke-width:3;" /><line x1="225" y1="0" x2="225" y2="20" style="stroke:#f5e48b;stroke-width:3;" /><line x1="226" y1="0" x2="226" y2="20" style="stroke:#f5e48b;stroke-width:3;" /><line x1="227" y1="0" x2="227" y2="20" style="stroke:#f4e58b;stroke-width:3;" /><line x1="228" y1="0" x2="228" y2="20" style="stroke:#f4e58b;stroke-width:3;" /><line x1="229" y1="0" x2="229" y2="20" style="stroke:#f4e58b;stroke-width:3;" /><line x1="230" y1="0" x2="230" y2="20" style="stroke:#f3e58b;stroke-width:3;" /><line x1="231" y1="0" x2="231" y2="20" style="stroke:#f3e58b;stroke-width:3;" /><line x1="232" y1="0" x2="232" y2="20" style="stroke:#f2e58b;stroke-width:3;" /><line x1="233" y1="0" x2="233" y2="20" style="stroke:#f2e58b;stroke-width:3;" /><line x1="234" y1="0" x2="234" y2="20" style="stroke:#f2e68b;stroke-width:3;" /><line x1="235" y1="0" x2="235" y2="20" style="stroke:#f1e68b;stroke-width:3;" /><line x1="236" y1="0" x2="236" y2="20" style="stroke:#f1e68b;stroke-width:3;" /><line x1="237" y1="0" x2="237" y2="20" style="stroke:#f1e68b;stroke-width:3;" /><line x1="238" y1="0" x2="238" y2="20" style="stroke:#f0e68b;stroke-width:3;" /><line x1="239" y1="0" x2="239" y2="20" style="stroke:#f0e68b;stroke-width:3;" /><line x1="240" y1="0" x2="240" y2="20" style="stroke:#efe68b;stroke-width:3;" /><line x1="241" y1="0" x2="241" y2="20" style="stroke:#efe78b;stroke-width:3;" /><line x1="242" y1="0" x2="242" y2="20" style="stroke:#efe78b;stroke-width:3;" /><line x1="243" y1="0" x2="243" y2="20" style="stroke:#eee78b;stroke-width:3;" /><line x1="244" y1="0" x2="244" y2="20" style="stroke:#eee78b;stroke-width:3;" /><line x1="245" y1="0" x2="245" y2="20" style="stroke:#eee78b;stroke-width:3;" /><line x1="246" y1="0" x2="246" y2="20" style="stroke:#ede78b;stroke-width:3;" /><line x1="247" y1="0" x2="247" y2="20" style="stroke:#ede88b;stroke-width:3;" /><line x1="248" y1="0" x2="248" y2="20" style="stroke:#ece88b;stroke-width:3;" /><line x1="249" y1="0" x2="249" y2="20" style="stroke:#ece88b;stroke-width:3;" /><line x1="250" y1="0" x2="250" y2="20" style="stroke:#ece88b;stroke-width:3;" /><line x1="251" y1="0" x2="251" y2="20" style="stroke:#ebe88b;stroke-width:3;" /><line x1="252" y1="0" x2="252" y2="20" style="stroke:#ebe88b;stroke-width:3;" /><line x1="253" y1="0" x2="253" y2="20" style="stroke:#ebe88b;stroke-width:3;" /><line x1="254" y1="0" x2="254" y2="20" style="stroke:#eae98b;stroke-width:3;" /><line x1="255" y1="0" x2="255" y2="20" style="stroke:#eae98b;stroke-width:3;" /><line x1="256" y1="0" x2="256" y2="20" style="stroke:#eae98b;stroke-width:3;" /><line x1="257" y1="0" x2="257" y2="20" style="stroke:#e9e98b;stroke-width:3;" /><line x1="258" y1="0" x2="258" y2="20" style="stroke:#e9e98b;stroke-width:3;" /><line x1="259" y1="0" x2="259" y2="20" style="stroke:#e8e98b;stroke-width:3;" /><line x1="260" y1="0" x2="260" y2="20" style="stroke:#e8e98b;stroke-width:3;" /><line x1="261" y1="0" x2="261" y2="20" style="stroke:#e8ea8b;stroke-width:3;" /><line x1="262" y1="0" x2="262" y2="20" style="stroke:#e7ea8b;stroke-width:3;" /><line x1="263" y1="0" x2="263" y2="20" style="stroke:#e7ea8b;stroke-width:3;" /><line x1="264" y1="0" x2="264" y2="20" style="stroke:#e7ea8b;stroke-width:3;" /><line x1="265" y1="0" x2="265" y2="20" style="stroke:#e6ea8b;stroke-width:3;" /><line x1="266" y1="0" x2="266" y2="20" style="stroke:#e6ea8b;stroke-width:3;" /><line x1="267" y1="0" x2="267" y2="20" style="stroke:#e5eb8b;stroke-width:3;" /><line x1="268" y1="0" x2="268" y2="20" style="stroke:#e5eb8b;stroke-width:3;" /><line x1="269" y1="0" x2="269" y2="20" style="stroke:#e5eb8b;stroke-width:3;" /><line x1="270" y1="0" x2="270" y2="20" style="stroke:#e4eb8b;stroke-width:3;" /><line x1="271" y1="0" x2="271" y2="20" style="stroke:#e4eb8b;stroke-width:3;" /><line x1="272" y1="0" x2="272" y2="20" style="stroke:#e4eb8b;stroke-width:3;" /><line x1="273" y1="0" x2="273" y2="20" style="stroke:#e3eb8b;stroke-width:3;" /><line x1="274" y1="0" x2="274" y2="20" style="stroke:#e3ec8b;stroke-width:3;" /><line x1="275" y1="0" x2="275" y2="20" style="stroke:#e2ec8b;stroke-width:3;" /><line x1="276" y1="0" x2="276" y2="20" style="stroke:#e2ec8b;stroke-width:3;" /><line x1="277" y1="0" x2="277" y2="20" style="stroke:#e2ec8b;stroke-width:3;" /><line x1="278" y1="0" x2="278" y2="20" style="stroke:#e1ec8b;stroke-width:3;" /><line x1="279" y1="0" x2="279" y2="20" style="stroke:#e1ec8b;stroke-width:3;" /><line x1="280" y1="0" x2="280" y2="20" style="stroke:#e1ed8b;stroke-width:3;" /><line x1="281" y1="0" x2="281" y2="20" style="stroke:#e0ed8b;stroke-width:3;" /><line x1="282" y1="0" x2="282" y2="20" style="stroke:#e0ed8b;stroke-width:3;" /><line x1="283" y1="0" x2="283" y2="20" style="stroke:#dfed8b;stroke-width:3;" /><line x1="284" y1="0" x2="284" y2="20" style="stroke:#dfed8b;stroke-width:3;" /><line x1="285" y1="0" x2="285" y2="20" style="stroke:#dfed8b;stroke-width:3;" /><line x1="286" y1="0" x2="286" y2="20" style="stroke:#deed8b;stroke-width:3;" /><line x1="287" y1="0" x2="287" y2="20" style="stroke:#deee8b;stroke-width:3;" /><line x1="288" y1="0" x2="288" y2="20" style="stroke:#deee8b;stroke-width:3;" /><line x1="289" y1="0" x2="289" y2="20" style="stroke:#ddee8b;stroke-width:3;" /><line x1="290" y1="0" x2="290" y2="20" style="stroke:#ddee8b;stroke-width:3;" /><line x1="291" y1="0" x2="291" y2="20" style="stroke:#dcee8b;stroke-width:3;" /><line x1="292" y1="0" x2="292" y2="20" style="stroke:#dcee8b;stroke-width:3;" /><line x1="293" y1="0" x2="293" y2="20" style="stroke:#dcee8b;stroke-width:3;" /><line x1="294" y1="0" x2="294" y2="20" style="stroke:#dbef8b;stroke-width:3;" /><line x1="295" y1="0" x2="295" y2="20" style="stroke:#dbef8b;stroke-width:3;" /><line x1="296" y1="0" x2="296" y2="20" style="stroke:#dbef8b;stroke-width:3;" /><line x1="297" y1="0" x2="297" y2="20" style="stroke:#daef8b;stroke-width:3;" /><line x1="298" y1="0" x2="298" y2="20" style="stroke:#daef8b;stroke-width:3;" /><line x1="299" y1="0" x2="299" y2="20" style="stroke:#d9ef8b;stroke-width:3;" /><line x1="300" y1="0" x2="300" y2="20" style="stroke:#d9ef8b;stroke-width:3;" /><line x1="301" y1="0" x2="301" y2="20" style="stroke:#d8ef8a;stroke-width:3;" /><line x1="302" y1="0" x2="302" y2="20" style="stroke:#d7ef8a;stroke-width:3;" /><line x1="303" y1="0" x2="303" y2="20" style="stroke:#d7ee89;stroke-width:3;" /><line x1="304" y1="0" x2="304" y2="20" style="stroke:#d6ee89;stroke-width:3;" /><line x1="305" y1="0" x2="305" y2="20" style="stroke:#d5ee89;stroke-width:3;" /><line x1="306" y1="0" x2="306" y2="20" style="stroke:#d5ed88;stroke-width:3;" /><line x1="307" y1="0" x2="307" y2="20" style="stroke:#d4ed88;stroke-width:3;" /><line x1="308" y1="0" x2="308" y2="20" style="stroke:#d3ed87;stroke-width:3;" /><line x1="309" y1="0" x2="309" y2="20" style="stroke:#d2ec87;stroke-width:3;" /><line x1="310" y1="0" x2="310" y2="20" style="stroke:#d2ec86;stroke-width:3;" /><line x1="311" y1="0" x2="311" y2="20" style="stroke:#d1ec86;stroke-width:3;" /><line x1="312" y1="0" x2="312" y2="20" style="stroke:#d0eb86;stroke-width:3;" /><line x1="313" y1="0" x2="313" y2="20" style="stroke:#d0eb85;stroke-width:3;" /><line x1="314" y1="0" x2="314" y2="20" style="stroke:#cfeb85;stroke-width:3;" /><line x1="315" y1="0" x2="315" y2="20" style="stroke:#ceea84;stroke-width:3;" /><line x1="316" y1="0" x2="316" y2="20" style="stroke:#cdea84;stroke-width:3;" /><line x1="317" y1="0" x2="317" y2="20" style="stroke:#cdea83;stroke-width:3;" /><line x1="318" y1="0" x2="318" y2="20" style="stroke:#cce983;stroke-width:3;" /><line x1="319" y1="0" x2="319" y2="20" style="stroke:#cbe983;stroke-width:3;" /><line x1="320" y1="0" x2="320" y2="20" style="stroke:#cae982;stroke-width:3;" /><line x1="321" y1="0" x2="321" y2="20" style="stroke:#cae882;stroke-width:3;" /><line x1="322" y1="0" x2="322" y2="20" style="stroke:#c9e881;stroke-width:3;" /><line x1="323" y1="0" x2="323" y2="20" style="stroke:#c8e881;stroke-width:3;" /><line x1="324" y1="0" x2="324" y2="20" style="stroke:#c8e880;stroke-width:3;" /><line x1="325" y1="0" x2="325" y2="20" style="stroke:#c7e780;stroke-width:3;" /><line x1="326" y1="0" x2="326" y2="20" style="stroke:#c6e780;stroke-width:3;" /><line x1="327" y1="0" x2="327" y2="20" style="stroke:#c5e77f;stroke-width:3;" /><line x1="328" y1="0" x2="328" y2="20" style="stroke:#c5e67f;stroke-width:3;" /><line x1="329" y1="0" x2="329" y2="20" style="stroke:#c4e67e;stroke-width:3;" /><line x1="330" y1="0" x2="330" y2="20" style="stroke:#c3e67e;stroke-width:3;" /><line x1="331" y1="0" x2="331" y2="20" style="stroke:#c2e57d;stroke-width:3;" /><line x1="332" y1="0" x2="332" y2="20" style="stroke:#c2e57d;stroke-width:3;" /><line x1="333" y1="0" x2="333" y2="20" style="stroke:#c1e57d;stroke-width:3;" /><line x1="334" y1="0" x2="334" y2="20" style="stroke:#c0e47c;stroke-width:3;" /><line x1="335" y1="0" x2="335" y2="20" style="stroke:#c0e47c;stroke-width:3;" /><line x1="336" y1="0" x2="336" y2="20" style="stroke:#bfe47b;stroke-width:3;" /><line x1="337" y1="0" x2="337" y2="20" style="stroke:#bee37b;stroke-width:3;" /><line x1="338" y1="0" x2="338" y2="20" style="stroke:#bde37a;stroke-width:3;" /><line x1="339" y1="0" x2="339" y2="20" style="stroke:#bde37a;stroke-width:3;" /><line x1="340" y1="0" x2="340" y2="20" style="stroke:#bce279;stroke-width:3;" /><line x1="341" y1="0" x2="341" y2="20" style="stroke:#bbe279;stroke-width:3;" /><line x1="342" y1="0" x2="342" y2="20" style="stroke:#bae279;stroke-width:3;" /><line x1="343" y1="0" x2="343" y2="20" style="stroke:#bae178;stroke-width:3;" /><line x1="344" y1="0" x2="344" y2="20" style="stroke:#b9e178;stroke-width:3;" /><line x1="345" y1="0" x2="345" y2="20" style="stroke:#b8e177;stroke-width:3;" /><line x1="346" y1="0" x2="346" y2="20" style="stroke:#b8e077;stroke-width:3;" /><line x1="347" y1="0" x2="347" y2="20" style="stroke:#b7e076;stroke-width:3;" /><line x1="348" y1="0" x2="348" y2="20" style="stroke:#b6e076;stroke-width:3;" /><line x1="349" y1="0" x2="349" y2="20" style="stroke:#b5df76;stroke-width:3;" /><line x1="350" y1="0" x2="350" y2="20" style="stroke:#b5df75;stroke-width:3;" /><line x1="351" y1="0" x2="351" y2="20" style="stroke:#b4df75;stroke-width:3;" /><line x1="352" y1="0" x2="352" y2="20" style="stroke:#b3df74;stroke-width:3;" /><line x1="353" y1="0" x2="353" y2="20" style="stroke:#b3de74;stroke-width:3;" /><line x1="354" y1="0" x2="354" y2="20" style="stroke:#b2de73;stroke-width:3;" /><line x1="355" y1="0" x2="355" y2="20" style="stroke:#b1de73;stroke-width:3;" /><line x1="356" y1="0" x2="356" y2="20" style="stroke:#b0dd73;stroke-width:3;" /><line x1="357" y1="0" x2="357" y2="20" style="stroke:#b0dd72;stroke-width:3;" /><line x1="358" y1="0" x2="358" y2="20" style="stroke:#afdd72;stroke-width:3;" /><line x1="359" y1="0" x2="359" y2="20" style="stroke:#aedc71;stroke-width:3;" /><line x1="360" y1="0" x2="360" y2="20" style="stroke:#addc71;stroke-width:3;" /><line x1="361" y1="0" x2="361" y2="20" style="stroke:#addc70;stroke-width:3;" /><line x1="362" y1="0" x2="362" y2="20" style="stroke:#acdb70;stroke-width:3;" /><line x1="363" y1="0" x2="363" y2="20" style="stroke:#abdb70;stroke-width:3;" /><line x1="364" y1="0" x2="364" y2="20" style="stroke:#abdb6f;stroke-width:3;" /><line x1="365" y1="0" x2="365" y2="20" style="stroke:#aada6f;stroke-width:3;" /><line x1="366" y1="0" x2="366" y2="20" style="stroke:#a9da6e;stroke-width:3;" /><line x1="367" y1="0" x2="367" y2="20" style="stroke:#a8da6e;stroke-width:3;" /><line x1="368" y1="0" x2="368" y2="20" style="stroke:#a8d96d;stroke-width:3;" /><line x1="369" y1="0" x2="369" y2="20" style="stroke:#a7d96d;stroke-width:3;" /><line x1="370" y1="0" x2="370" y2="20" style="stroke:#a6d96d;stroke-width:3;" /><line x1="371" y1="0" x2="371" y2="20" style="stroke:#a5d86c;stroke-width:3;" /><line x1="372" y1="0" x2="372" y2="20" style="stroke:#a5d86c;stroke-width:3;" /><line x1="373" y1="0" x2="373" y2="20" style="stroke:#a4d86b;stroke-width:3;" /><line x1="374" y1="0" x2="374" y2="20" style="stroke:#a3d76b;stroke-width:3;" /><line x1="375" y1="0" x2="375" y2="20" style="stroke:#a3d76a;stroke-width:3;" /><line x1="376" y1="0" x2="376" y2="20" style="stroke:#a2d76a;stroke-width:3;" /><line x1="377" y1="0" x2="377" y2="20" style="stroke:#a1d669;stroke-width:3;" /><line x1="378" y1="0" x2="378" y2="20" style="stroke:#a0d669;stroke-width:3;" /><line x1="379" y1="0" x2="379" y2="20" style="stroke:#a0d669;stroke-width:3;" /><line x1="380" y1="0" x2="380" y2="20" style="stroke:#9fd568;stroke-width:3;" /><line x1="381" y1="0" x2="381" y2="20" style="stroke:#9ed568;stroke-width:3;" /><line x1="382" y1="0" x2="382" y2="20" style="stroke:#9ed567;stroke-width:3;" /><line x1="383" y1="0" x2="383" y2="20" style="stroke:#9dd567;stroke-width:3;" /><line x1="384" y1="0" x2="384" y2="20" style="stroke:#9cd466;stroke-width:3;" /><line x1="385" y1="0" x2="385" y2="20" style="stroke:#9bd466;stroke-width:3;" /><line x1="386" y1="0" x2="386" y2="20" style="stroke:#9bd466;stroke-width:3;" /><line x1="387" y1="0" x2="387" y2="20" style="stroke:#9ad365;stroke-width:3;" /><line x1="388" y1="0" x2="388" y2="20" style="stroke:#99d365;stroke-width:3;" /><line x1="389" y1="0" x2="389" y2="20" style="stroke:#98d364;stroke-width:3;" /><line x1="390" y1="0" x2="390" y2="20" style="stroke:#98d264;stroke-width:3;" /><line x1="391" y1="0" x2="391" y2="20" style="stroke:#97d263;stroke-width:3;" /><line x1="392" y1="0" x2="392" y2="20" style="stroke:#96d263;stroke-width:3;" /><line x1="393" y1="0" x2="393" y2="20" style="stroke:#96d163;stroke-width:3;" /><line x1="394" y1="0" x2="394" y2="20" style="stroke:#95d162;stroke-width:3;" /><line x1="395" y1="0" x2="395" y2="20" style="stroke:#94d162;stroke-width:3;" /><line x1="396" y1="0" x2="396" y2="20" style="stroke:#93d061;stroke-width:3;" /><line x1="397" y1="0" x2="397" y2="20" style="stroke:#93d061;stroke-width:3;" /><line x1="398" y1="0" x2="398" y2="20" style="stroke:#92d060;stroke-width:3;" /><line x1="399" y1="0" x2="399" y2="20" style="stroke:#91cf60;stroke-width:3;" /><line x1="400" y1="0" x2="400" y2="20" style="stroke:#90cf60;stroke-width:3;" /><line x1="401" y1="0" x2="401" y2="20" style="stroke:#8fce60;stroke-width:3;" /><line x1="402" y1="0" x2="402" y2="20" style="stroke:#8ece5f;stroke-width:3;" /><line x1="403" y1="0" x2="403" y2="20" style="stroke:#8dcd5f;stroke-width:3;" /><line x1="404" y1="0" x2="404" y2="20" style="stroke:#8bcd5f;stroke-width:3;" /><line x1="405" y1="0" x2="405" y2="20" style="stroke:#8acc5f;stroke-width:3;" /><line x1="406" y1="0" x2="406" y2="20" style="stroke:#89cc5f;stroke-width:3;" /><line x1="407" y1="0" x2="407" y2="20" style="stroke:#88cb5f;stroke-width:3;" /><line x1="408" y1="0" x2="408" y2="20" style="stroke:#87ca5e;stroke-width:3;" /><line x1="409" y1="0" x2="409" y2="20" style="stroke:#85ca5e;stroke-width:3;" /><line x1="410" y1="0" x2="410" y2="20" style="stroke:#84c95e;stroke-width:3;" /><line x1="411" y1="0" x2="411" y2="20" style="stroke:#83c95e;stroke-width:3;" /><line x1="412" y1="0" x2="412" y2="20" style="stroke:#82c85e;stroke-width:3;" /><line x1="413" y1="0" x2="413" y2="20" style="stroke:#81c85e;stroke-width:3;" /><line x1="414" y1="0" x2="414" y2="20" style="stroke:#7fc75d;stroke-width:3;" /><line x1="415" y1="0" x2="415" y2="20" style="stroke:#7ec75d;stroke-width:3;" /><line x1="416" y1="0" x2="416" y2="20" style="stroke:#7dc65d;stroke-width:3;" /><line x1="417" y1="0" x2="417" y2="20" style="stroke:#7cc55d;stroke-width:3;" /><line x1="418" y1="0" x2="418" y2="20" style="stroke:#7bc55d;stroke-width:3;" /><line x1="419" y1="0" x2="419" y2="20" style="stroke:#79c45d;stroke-width:3;" /><line x1="420" y1="0" x2="420" y2="20" style="stroke:#78c45d;stroke-width:3;" /><line x1="421" y1="0" x2="421" y2="20" style="stroke:#77c35c;stroke-width:3;" /><line x1="422" y1="0" x2="422" y2="20" style="stroke:#76c35c;stroke-width:3;" /><line x1="423" y1="0" x2="423" y2="20" style="stroke:#75c25c;stroke-width:3;" /><line x1="424" y1="0" x2="424" y2="20" style="stroke:#73c25c;stroke-width:3;" /><line x1="425" y1="0" x2="425" y2="20" style="stroke:#72c15c;stroke-width:3;" /><line x1="426" y1="0" x2="426" y2="20" style="stroke:#71c05c;stroke-width:3;" /><line x1="427" y1="0" x2="427" y2="20" style="stroke:#70c05b;stroke-width:3;" /><line x1="428" y1="0" x2="428" y2="20" style="stroke:#6fbf5b;stroke-width:3;" /><line x1="429" y1="0" x2="429" y2="20" style="stroke:#6dbf5b;stroke-width:3;" /><line x1="430" y1="0" x2="430" y2="20" style="stroke:#6cbe5b;stroke-width:3;" /><line x1="431" y1="0" x2="431" y2="20" style="stroke:#6bbe5b;stroke-width:3;" /><line x1="432" y1="0" x2="432" y2="20" style="stroke:#6abd5b;stroke-width:3;" /><line x1="433" y1="0" x2="433" y2="20" style="stroke:#69bd5a;stroke-width:3;" /><line x1="434" y1="0" x2="434" y2="20" style="stroke:#67bc5a;stroke-width:3;" /><line x1="435" y1="0" x2="435" y2="20" style="stroke:#66bc5a;stroke-width:3;" /><line x1="436" y1="0" x2="436" y2="20" style="stroke:#65bb5a;stroke-width:3;" /><line x1="437" y1="0" x2="437" y2="20" style="stroke:#64ba5a;stroke-width:3;" /><line x1="438" y1="0" x2="438" y2="20" style="stroke:#63ba5a;stroke-width:3;" /><line x1="439" y1="0" x2="439" y2="20" style="stroke:#61b959;stroke-width:3;" /><line x1="440" y1="0" x2="440" y2="20" style="stroke:#60b959;stroke-width:3;" /><line x1="441" y1="0" x2="441" y2="20" style="stroke:#5fb859;stroke-width:3;" /><line x1="442" y1="0" x2="442" y2="20" style="stroke:#5eb859;stroke-width:3;" /><line x1="443" y1="0" x2="443" y2="20" style="stroke:#5db759;stroke-width:3;" /><line x1="444" y1="0" x2="444" y2="20" style="stroke:#5bb759;stroke-width:3;" /><line x1="445" y1="0" x2="445" y2="20" style="stroke:#5ab659;stroke-width:3;" /><line x1="446" y1="0" x2="446" y2="20" style="stroke:#59b558;stroke-width:3;" /><line x1="447" y1="0" x2="447" y2="20" style="stroke:#58b558;stroke-width:3;" /><line x1="448" y1="0" x2="448" y2="20" style="stroke:#57b458;stroke-width:3;" /><line x1="449" y1="0" x2="449" y2="20" style="stroke:#55b458;stroke-width:3;" /><line x1="450" y1="0" x2="450" y2="20" style="stroke:#54b358;stroke-width:3;" /><line x1="451" y1="0" x2="451" y2="20" style="stroke:#53b358;stroke-width:3;" /><line x1="452" y1="0" x2="452" y2="20" style="stroke:#52b257;stroke-width:3;" /><line x1="453" y1="0" x2="453" y2="20" style="stroke:#51b257;stroke-width:3;" /><line x1="454" y1="0" x2="454" y2="20" style="stroke:#4fb157;stroke-width:3;" /><line x1="455" y1="0" x2="455" y2="20" style="stroke:#4eb057;stroke-width:3;" /><line x1="456" y1="0" x2="456" y2="20" style="stroke:#4db057;stroke-width:3;" /><line x1="457" y1="0" x2="457" y2="20" style="stroke:#4caf57;stroke-width:3;" /><line x1="458" y1="0" x2="458" y2="20" style="stroke:#4baf56;stroke-width:3;" /><line x1="459" y1="0" x2="459" y2="20" style="stroke:#49ae56;stroke-width:3;" /><line x1="460" y1="0" x2="460" y2="20" style="stroke:#48ae56;stroke-width:3;" /><line x1="461" y1="0" x2="461" y2="20" style="stroke:#47ad56;stroke-width:3;" /><line x1="462" y1="0" x2="462" y2="20" style="stroke:#46ad56;stroke-width:3;" /><line x1="463" y1="0" x2="463" y2="20" style="stroke:#45ac56;stroke-width:3;" /><line x1="464" y1="0" x2="464" y2="20" style="stroke:#43ab55;stroke-width:3;" /><line x1="465" y1="0" x2="465" y2="20" style="stroke:#42ab55;stroke-width:3;" /><line x1="466" y1="0" x2="466" y2="20" style="stroke:#41aa55;stroke-width:3;" /><line x1="467" y1="0" x2="467" y2="20" style="stroke:#40aa55;stroke-width:3;" /><line x1="468" y1="0" x2="468" y2="20" style="stroke:#3fa955;stroke-width:3;" /><line x1="469" y1="0" x2="469" y2="20" style="stroke:#3ea955;stroke-width:3;" /><line x1="470" y1="0" x2="470" y2="20" style="stroke:#3ca854;stroke-width:3;" /><line x1="471" y1="0" x2="471" y2="20" style="stroke:#3ba854;stroke-width:3;" /><line x1="472" y1="0" x2="472" y2="20" style="stroke:#3aa754;stroke-width:3;" /><line x1="473" y1="0" x2="473" y2="20" style="stroke:#39a654;stroke-width:3;" /><line x1="474" y1="0" x2="474" y2="20" style="stroke:#38a654;stroke-width:3;" /><line x1="475" y1="0" x2="475" y2="20" style="stroke:#36a554;stroke-width:3;" /><line x1="476" y1="0" x2="476" y2="20" style="stroke:#35a554;stroke-width:3;" /><line x1="477" y1="0" x2="477" y2="20" style="stroke:#34a453;stroke-width:3;" /><line x1="478" y1="0" x2="478" y2="20" style="stroke:#33a453;stroke-width:3;" /><line x1="479" y1="0" x2="479" y2="20" style="stroke:#32a353;stroke-width:3;" /><line x1="480" y1="0" x2="480" y2="20" style="stroke:#30a353;stroke-width:3;" /><line x1="481" y1="0" x2="481" y2="20" style="stroke:#2fa253;stroke-width:3;" /><line x1="482" y1="0" x2="482" y2="20" style="stroke:#2ea253;stroke-width:3;" /><line x1="483" y1="0" x2="483" y2="20" style="stroke:#2da152;stroke-width:3;" /><line x1="484" y1="0" x2="484" y2="20" style="stroke:#2ca052;stroke-width:3;" /><line x1="485" y1="0" x2="485" y2="20" style="stroke:#2aa052;stroke-width:3;" /><line x1="486" y1="0" x2="486" y2="20" style="stroke:#299f52;stroke-width:3;" /><line x1="487" y1="0" x2="487" y2="20" style="stroke:#289f52;stroke-width:3;" /><line x1="488" y1="0" x2="488" y2="20" style="stroke:#279e52;stroke-width:3;" /><line x1="489" y1="0" x2="489" y2="20" style="stroke:#269e51;stroke-width:3;" /><line x1="490" y1="0" x2="490" y2="20" style="stroke:#249d51;stroke-width:3;" /><line x1="491" y1="0" x2="491" y2="20" style="stroke:#239d51;stroke-width:3;" /><line x1="492" y1="0" x2="492" y2="20" style="stroke:#229c51;stroke-width:3;" /><line x1="493" y1="0" x2="493" y2="20" style="stroke:#219b51;stroke-width:3;" /><line x1="494" y1="0" x2="494" y2="20" style="stroke:#209b51;stroke-width:3;" /><line x1="495" y1="0" x2="495" y2="20" style="stroke:#1e9a50;stroke-width:3;" /><line x1="496" y1="0" x2="496" y2="20" style="stroke:#1d9a50;stroke-width:3;" /><line x1="497" y1="0" x2="497" y2="20" style="stroke:#1c9950;stroke-width:3;" /><line x1="498" y1="0" x2="498" y2="20" style="stroke:#1b9950;stroke-width:3;" /><line x1="499" y1="0" x2="499" y2="20" style="stroke:#1a9850;stroke-width:3;" /><text x="0" y="35">0.2</text><text x="500" y="35" style="text-anchor:end;">0.9</text></svg>




```python
# Create a regular PCA model 
pca = decomposition.PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(features)

plt.scatter(x=reduced_data_pca[:,0], y=reduced_data_pca[:,1], c=labels.accuracy, alpha=0.6)
plt.xlabel('1st principal component')
plt.ylabel('2nd principal component')
plt.colorbar()
plt.title("PCA with user accuracy")
plt.show()
```


![png](images/output_28_0.png)


In the 2-component PCA scatter, we see two clusters. The left cluster is smaller and is composed of a subset of all high-performing users. In the center, the cluster contains a majority of users and is mixed between high and low-performing users. This is an indication that low-performing will be difficult to isolate from others.

#### SVM
##### Regression
To model this data, we can use supervised machine learning. One popular class of algorithm is support vector machine (SVM). For classification, SVM selects hyperplanes that minimize error by maximizing the margin between classes of data. For regression, it works similarilty: a hyper plane is fit along the data. If the data is not linear, a kernel function is used to transform the data. Our user accuracy label is continuous, so we will use regression.

SVM regression has two major hyper parameters: $C$, the penalty for error, and $epsilon$, the margin of tolerance for errors. In order to choose values for the hyper parameters, we can test out a range of options. Here, we use the scikit-learn SVM regression implementation with a radial-basis kernel, $epsilon$ of $0.1$, and a range of $C$ values from $0.01$ to $1.0$. The tests are done with 3-fold cross validation.


```python
c_values = np.linspace(0.01, 1, 50)
svr = svm.SVR(epsilon=0.1)
svr_scores = []
svr_err = []
for c in c_values:
    svr.set_params(C=c)
    cv = model_selection.cross_val_score(svr, features, labels.accuracy, cv=3)
    svr_scores.append(np.mean(cv))
    svr_err.append(np.std(cv))
print('Max C:', c_values[np.argmax(svr_scores)])
plt.xlabel('C')
plt.ylabel('cv r^2 score')
plt.errorbar(c_values, svr_scores, label='svr', yerr=svr_err, ecolor='grey')
plt.legend(loc='lower right')
plt.show()
```

    Max C: 0.191836734694



![png](images/output_30_1.png)


The regressor had the best performance around $C=0.2$ We can rerun a similar test on possible $epsilon$ values.


```python
epsilon_values = np.linspace(0.00001, 0.5, 50)
svr = svm.SVR(C=0.2)
svr_scores = []
svr_err = []
for epsilon in epsilon_values:
    svr.set_params(epsilon=epsilon)
    cv = model_selection.cross_val_score(svr, features, labels.accuracy, cv=3)
    svr_scores.append(np.mean(cv))
    svr_err.append(np.std(cv))
print('Max epsilon:', epsilon_values[np.argmax(svr_scores)])
plt.xlabel('epsilon')
plt.ylabel('cv r^2 score')
plt.errorbar(epsilon_values, svr_scores, label='svr', yerr=svr_err, ecolor='grey')
plt.legend(loc='lower right')
plt.show()
```

    Max epsilon: 0.0510293877551



![png](images/output_32_1.png)


The regessor had the best performance around $epsilon = 0.05$. We've seen how the parameters impact the model individualy, but not how they interact together. In order to optimize the model with both of them, we can use a grid search to test out each combination of parameters. In addition to the radial-basis function, we also test a polynomial kernel. We also increase the cross validation to 10-fold.


```python
parameters ={
    'C': np.power(10.0,range(-2,3)), 'epsilon': np.power(10.0,range(-4,-1)), 'kernel': ('poly', 'rbf')}
svr = svm.SVR()
clf = model_selection.GridSearchCV(svr, parameters, n_jobs=-1)
scores = model_selection.cross_val_score(clf, features, labels.accuracy, cv=10)
fig = plt.figure()
sns.boxplot(x=scores)
plt.xlabel('cv r^2 score')
fig.set_size_inches(5,1.5)
print("r^2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    r^2: 0.30 (+/- 0.19)



![png](images/output_34_1.png)


We received an $r^2$ score of 0.3 with a 95% confidence interval between 0.1 and 0.5. The $r^2$ score is an indication of how well the actual test data fitted the regression. A score of 1.0 would indicate a perfect fit. Our score indicates that the majority of the variation in accuracy is not explained by the model.

##### Classification

Because the SVM is struggling to predict accuracy as a continuous variable, we can ease its task. Using binning, we can discretize accuracy into categories. Here, we halve our data into two categories: users below median accuracy, and users above. To predict which group a user belongs to, we can use an SVM classifier. Here, we score the classifier's predictions using gridsearch on $C$ and kernel type with 10-fold cross validation.


```python
parameters = {'C': np.power(10.0,range(-2,3)), 'kernel': ('poly', 'rbf')}
binned_accuracy = (labels.accuracy < labels.accuracy.median()).apply(lambda x: 1 if x else 0)
svc = svm.SVC()
clf = model_selection.GridSearchCV(svc, parameters, n_jobs=-1)
scores = model_selection.cross_val_score(clf, features, binned_accuracy, cv=10)
fig = plt.figure()
sns.boxplot(x=scores)
plt.xlim(0,1)
fig.set_size_inches(7,1.5)
plt.xlabel('cv score')
print("cv: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

    cv: 0.68 (+/- 0.17)



![png](images/output_36_1.png)


On the binned data, the SVM correctly classifies about 70% of the users. Since the data is evenly distributed between binary categories, a random classification would result in 50% being correct. To understand where the classifier makes mistakes, we can view the confusion matrix.


```python
f_train, f_test, a_train, a_test = model_selection.train_test_split(features, binned_accuracy, test_size=0.5)
clf.fit(f_train, a_train)
metrics.confusion_matrix(a_test, clf.predict(f_test))
```




    array([[55, 23],
           [22, 54]])



As shown in the confusion matrix, the number of false positives and false negatives are approximately equal, as are the number of true positives and true negatives. By altering parameters of the model, we lean it more towards one of these error types. The receiver operating characteristic shows us the trade off between selecting all low-performing users users (0 false negative) and avoiding the selection of high-performing users by mistake (0 false positive).


```python
f_train, f_test, a_train, a_test = model_selection.train_test_split(features, binned_accuracy, test_size=0.5)
a_score = svm.SVC().fit(f_train, a_train).decision_function(f_test)
fpr, tpr, _ = metrics.roc_curve(a_test, a_score)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr,tpr, color='orange', label='ROC curve')
plt.plot([0,1],[0,1], color='b', linestyle='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")
print('area under curve:', roc_auc)
```

    area under curve: 0.785593792173



![png](images/output_40_1.png)


A perfect model would reach the top left corner of the graph and have 100% area under curve. Our model has around 80% area under curve.

##### Feature coefficients
SVM also allows us to look at the weights it gave to each feature, which makes the prediction more transparent.


```python
clf = svm.SVC(kernel='linear')
clf.fit(features, binned_accuracy)
coef = clf.coef_.ravel()
top_pos = np.argsort(coef)[::-1]
print('Top features leading to classification as poor performing:')
for i, pos in enumerate(top_pos[:5]):
    print(i+1, feature_cols[pos], coef[pos])
print('\nTop features leading to classification as high performing:')
for i, pos in enumerate(top_pos[-5:][::-1]):
    print(i+1, feature_cols[pos], coef[pos])
```

    Top features leading to classification as poor performing:
    1 LowLevelEvent_keydown_per_pan_std 1.12946853427
    2 PanoId_Changed_per_pan_std 0.931386498261
    3 ContextMenu_RadioChange_per_pan_std 0.929422110731
    4 PopUpShow_CheckBothSides_per_pan_mean 0.91432620227
    5 Click_ModeSwitch_NoCurbRamp_total 0.829416588987
    
    Top features leading to classification as high performing:
    1 TaskEnd_total -1.17914680012
    2 PanoId_Changed_per_pan_mean -1.15446703378
    3 ModeSwitch_CurbRamp_per_pan_mean -0.921973779003
    4 ModeSwitch_Walk_per_pan_mean -0.918854969043
    5 TaskEnd_per_pan_mean -0.912891970308

