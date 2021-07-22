
<center><h1>Title: Credit Card Fraud Detection Using Deep Learning.</h2></center>
<article>
<p>Credit card payment is a popular mode of online transaction. It is one of the simplest and easiest mode of payment
across the internet. However, with growing popularity of credit card transactions, there is an exponential growth in fraudulent
payments. Every year we lose billions of dollars due to fraudulent acts. These activities look like a genuine transaction; hence,
simple pattern techniques and less complex methods don't not go to work to minimize the fraudulent act and minimize the chaos
we are proposing a model consisting of Deep Learning Algorithm of Convolutional Neural Network. Theseapproache is proved to decrease the false alarm rates and increase the fraud detection rate and expected to be more efficient than other relevant algorithms.</p>
<article>

 <h2>DataSet</h2>
  <article><p>The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.</p><article>
    
<h3>Note:-Original and Filtered data is Provided in Repo</h3>
    
<h3>Programming Language : Python 3.8.10</h3>
<h3>Code Editor : Jupyter Notebook, VS Code </h3>
  
  
  <h2>Libraries/Modules Used</h2>
  <ol>
  <li>Tensorflow 2.0</li>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Streamlit</li>  
</ol>
  
<h2>File Structure</h2>
    <ol>
  <li>Resource</li>
      <ul>
        <li>Study Paper</li>
      </ul>
  <li>Code</li>
      <ul>
      <li>part1.ipynb</li>
      <li>part2.ipynb</li>
      <li>app2.py</li>
      <li>model.h5</li>  
    </ul>
  <li>Dset</li>
      <ul>
      <li>Creditcard.csv</li>
      <li>Dataset.csv</li>
    </ul>
</ol>
    
 
    
