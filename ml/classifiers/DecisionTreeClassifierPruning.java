package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * Decision tree classifier that supports multiclass classification.
 * 
 * The classifier handles non-binary features, however, when deciding on splits it
 * considers splitting by comparing zero vs. non-zero for each feature.
 * 
 * @author dkauchak
 *
 */
public class DecisionTreeClassifierPruning implements Classifier{
	//private DataSet allData;
	private HashMap<Integer,String> featureMap;
	private Set<Integer> featureIndices;
	private DecisionTreeNode decisionTree;
	private int depthMax = Integer.MAX_VALUE;
    private double errorThreshold = 0.01;
	
	public void train(DataSet data) {
		if( data.getData().size() == 0 ){
			throw new RuntimeException("Tried to train without any data");
		}
		
		featureMap = data.getFeatureMap();
		featureIndices = data.getAllFeatureIndices();
		decisionTree = buildTree(data.getData(), new HashSet<Integer>(), depthMax);
        decisionTree = pruneTree(decisionTree, data); 
	}
	
	/**
	 * Set the maximum height of the tree to be learned
	 * 
	 * @param depthMax the max depth of the tree
	 */
	public void setDepthLimit(int depthMax){
		this.depthMax = depthMax;
	}
	
	/**
	 * Helper method for building the decision tree.
	 * 
	 * @param currentData the data (non-empty) to build the tree over
	 * @param usedFeatures the features that have been used already
	 * @param depthLimit the maximum depth we can build this tree
	 * @return the learned decision tree
	 */
	private DecisionTreeNode buildTree(ArrayList<Example> currentData, HashSet<Integer> usedFeatures, int depthLimit){
		DataMajority majority = getMajorityLabel(currentData);
				
		// base cases:
		// 1. they're all the same label
		// 2. we're out of features to examine
		if( majority.majorityCount == currentData.size() ||
			usedFeatures.size() == featureIndices.size() ||
			depthLimit == 0){
			return new DecisionTreeNode(majority.majorityLabel, majority.confidence);
		}else{
			// check if all examples have the same features
					
			// find the best feature that hasn't been used yet to split on
			int bestFeature = getBestFeatureIndex(currentData, usedFeatures);
			
			// bestFeature != -1
			// split on the best feature
			ArrayList<Example>[] splits = splitData(currentData, bestFeature);
			
			// create a new decision tree node
			DecisionTreeNode node = new DecisionTreeNode(bestFeature);
			
			HashSet<Integer> featureCopy = (HashSet<Integer>)usedFeatures.clone();
			featureCopy.add(bestFeature);
			
			// left branch
			if( splits[0].size() == 0 ){
				node.setLeft(new DecisionTreeNode(majority.majorityLabel, majority.confidence));
			}else{
				node.setLeft(buildTree(splits[0],featureCopy, depthLimit-1));
			}
			
			// right branch
			if( splits[1].size() == 0 ){
				node.setRight(new DecisionTreeNode(majority.majorityLabel, majority.confidence));
			}else{
				node.setRight(buildTree(splits[1], featureCopy, depthLimit-1));
			}
			
			return node;
		}
	}
	
	/**
	 * Get the best feature to split on based on training error.
	 * 
	 * @param currentData the current set of examples
	 * @param usedFeatures which features have been used already and are NOT eligible for splitting on
	 * @return the index of the best feature
	 */
	private int getBestFeatureIndex(ArrayList<Example> currentData, HashSet<Integer> usedFeatures){
		int bestFeature = -1;
		double bestFeatureScore = 1.0; // lower is better for now
		
		for( int featureIndex: featureIndices){
			if( !usedFeatures.contains(featureIndex) ){
				double error = averageTrainingError(currentData, featureIndex);
									
				if( error < bestFeatureScore ||
					(error == bestFeatureScore && featureIndex < bestFeature )){
					bestFeatureScore = error;
					bestFeature = featureIndex;
				}
			}
		}
		
		return bestFeature;
	}
	
	/**
	 * Get the average training error on this data set if we split on featureIndex
	 * 
	 * @param data the current data
	 * @param featureIndex the feature we're considering splitting on
	 * @return the error
	 */
	private double averageTrainingError(ArrayList<Example> data, int featureIndex){		
		ArrayList<Example>[] splits = splitData(data, featureIndex);
		
		int leftCount = splits[0].size() > 0 ? getMajorityLabel(splits[0]).majorityCount : 0;
		int rightCount = splits[1].size() > 0 ? getMajorityLabel(splits[1]).majorityCount : 0;
		
		double accuracy = (leftCount+rightCount)/(double)data.size();
		return 1-accuracy;
	}
	
	/**
	 * Split the data based on featureIndex
	 * 
	 * @param data the data to be split
	 * @param featureIndex the feature to split on
	 * @return the split of the data.  Entry 0 is the left branch data and entry 1 the right branch data.
	 */
	private ArrayList<Example>[] splitData(ArrayList<Example> data, int featureIndex){
		// split the data based on this feature
		ArrayList<Example>[] splits = new ArrayList[2];
		splits[0] = new ArrayList<Example>();
		splits[1] = new ArrayList<Example>();
				
		for( Example d: data){
			double value = d.getFeature(featureIndex);
			
			if( value == DecisionTreeNode.LEFT_BRANCH ){
				splits[0].add(d);
			}else{
				splits[1].add(d);
			}
		}
		
		return splits;
	}
	
	public String toString(){
		return decisionTree.treeString(featureMap);
	}
	
	/**
	 * given the data, calculate the majority label and how many times it occurs in the data
	 * 
	 * @param data
	 * @return majority information from the data
	 */
	private DataMajority getMajorityLabel(ArrayList<Example> data){
		HashMapCounter<Double> counter = new HashMapCounter<Double>();
		
		for( Example d: data ){
			counter.increment(d.getLabel());
		}
		
		double maxLabel = 0.0;
		int maxCount = -1;
		
		for( Double key: counter.keySet() ){
			if( counter.get(key) > maxCount ){
				maxCount = counter.get(key);
				maxLabel = key;
			}
		}
		
		return new DataMajority(maxLabel, maxCount, ((double)maxCount)/data.size());
	}
		
	@Override
	public double classify(Example example) {
		return findLeaf(example).prediction();
	}
	
	@Override
	public double confidence(Example example) {
		return findLeaf(example).confidence();
	}
	
	/**
	 * Figure out which leaf this example falls into
	 * 
	 * @param example
	 * @return the leaf node
	 */
	private DecisionTreeNode findLeaf(Example example){
		DecisionTreeNode current = decisionTree;
		
		while( !current.isLeaf() ){
			int feature = current.getFeatureIndex();
			
			if( example.getFeature(feature) == DecisionTreeNode.LEFT_BRANCH ){
				// go left
				current = current.getLeft();
			}else{
				current = current.getRight();
			}
		}
		
		return current;
	}
		
	/**
	 * A container class to allow us to return multiple values when calculting
	 * the majority label from a collection of data.
	 * 
	 * @author dkauchak
	 *
	 */
	private class DataMajority{
		public double majorityLabel;
		public int majorityCount;
		public double confidence;
		
		public DataMajority(double majorityLabel, int majorityCount, double confidence){
			this.majorityLabel = majorityLabel;
			this.majorityCount = majorityCount;
			this.confidence = confidence;
		}
	}
    
	private DecisionTreeNode pruneTree(DecisionTreeNode node, DataSet data) {
		if (node == null || node.isLeaf()) {
			return node;
		}
		 // Recursively prune left and right children and update references
		 node.setLeft(pruneTree(node.getLeft(), data));
		 node.setRight(pruneTree(node.getRight(), data));
	 
		 CrossValidationSet cvs = new CrossValidationSet(data, 2);
		 double prePruneError = calculateError(cvs.getValidationSet(0).getTest().getData());
	 
		 // Create a temporary leaf node for pruning
		 DataMajority majority = getMajorityLabel(data.getData());
		 DecisionTreeNode leafNode = new DecisionTreeNode(majority.majorityLabel, majority.confidence);
	 
		 // Temporarily replace node with leafNode for post-prune error calculation
		 DecisionTreeNode originalNode = new DecisionTreeNode(node.getFeatureIndex());
		 originalNode.setLeft(node.getLeft());
		 originalNode.setRight(node.getRight());
	 
		 // Replace current node with leafNode
		 node = leafNode;
		 double postPruneError = calculateError(cvs.getValidationSet(1).getTest().getData());
	 
		 // Determine whether to prune or not
		 if (postPruneError <= prePruneError) {
			 // Pruning is beneficial, return the new leaf node
			 return leafNode;
		 }
		  else {
			 // Pruning is not beneficial, return the original node
			 return originalNode;
		 }
	 }

    private double calculateError(ArrayList<Example> examples) {
        double incorrect = 0;

        for (Example example : examples) {
            double prediction = classify(example);
            if (prediction != example.getLabel()) {
                incorrect++;
            }
        }

        return incorrect / examples.size();
    }
	public static void main(String[] args) {
  
        DataSet dataSet = new DataSet("/Users/virenjain/Documents/CS_158/ml-final/ml/utils/titanic.csv", 0); 
		int k = 5; 
		CrossValidationSet cvs = new CrossValidationSet(dataSet, k);
		double totalAccuracy = 0;
	
		for (int i = 0; i < k; i++) {
			DataSetSplit split = cvs.getValidationSet(i);
			DecisionTreeClassifierPruning classifier = new DecisionTreeClassifierPruning();
			classifier.train(split.getTrain());
	
			int correct = 0;
			for (Example example : split.getTest().getData()) {
				double predicted = classifier.classify(example);
				if (predicted == example.getLabel()) {
					correct++;
				}
			}
	
			double accuracy = (double) correct / split.getTest().getData().size();
			totalAccuracy += accuracy;
			System.out.println("Accuracy for fold " + (i+1) + ": " + accuracy);
		}
	
		double averageAccuracy = totalAccuracy / k;
		System.out.println("Average Accuracy: " + averageAccuracy);

		double totalAccuracy2 = 0;
		for (int i = 0; i < k; i++) {
			DataSetSplit split = cvs.getValidationSet(i);
			DecisionTreeClassifier classifier2 = new DecisionTreeClassifier();
			classifier2.train(split.getTrain());
	
			int correct = 0;
			for (Example example : split.getTest().getData()) {
				double predicted = classifier2.classify(example);
				if (predicted == example.getLabel()) {
					correct++;
				}
			}
	
			double accuracy = (double) correct / split.getTest().getData().size();
			totalAccuracy2 += accuracy;
			System.out.println("Accuracy for fold " + (i+1) + ": " + accuracy);
		}
	
		double averageAccuracy2 = totalAccuracy2 / k;
		System.out.println("Average Accuracy: " + averageAccuracy2);
	}
}