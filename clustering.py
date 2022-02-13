def clustering(X,Y):
    
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    import numpy as np
    import matplotlib.pyplot as plt
    from random import uniform
    from numpy import array,reshape,hstack,size

    # Configuration options
    num_samples_total = 1000
    
    # Remove this when using GPS coordinate inputs
    cluster_centers = [(32.30640,32.30650), (21.61450,21.61460),(25.00005,26.0003)]
    
    num_classes = len(cluster_centers)
    
    # Remove this when using GPS coordinate inputs
    X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)
    
    #dbscan configuration
    epsilon = 0.5
    min_samples = 5

    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
    labels = db.labels_
   
    no_clusters = len(np.unique(labels))-(1 if -1 in labels else 0)
    no_noise = np.sum(np.array(labels) == -1, axis=0)

    # remove noisy points (points labeled as noise by dbscan), make new array without these
    range_max = len(X)
    X = np.array([X[i] for i in range(range_max) if labels[i] != -1])
    labels = np.array([labels[i] for i in range(range_max) if labels[i] != -1])
 
    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)

    # Generate scatter plot for this data
    cluster_coords = []
    for i in range(len(np.unique(labels))):
        cluster_coords.append([])
    for i in range(0,len(labels)):
        cluster_coords[labels[i]].append(X[i])
    
    # Average cluster coordinates to find average centres
    # And Find centres coordinate of cluster with most points in it (aware this isnt the most elegant way)
    cluster_centres = []
    longest_index = 0
    length = []
    for i in range(len(np.unique(labels))):
        length.append(len(cluster_coords[i]))
        if length[i]==max(length):
            longest_index = i
        cluster_centres.append(np.average(cluster_coords[i],axis=0))
    print(cluster_centres[longest_index])
     
    # Plot points, with cluster with max number of points coloured red and other clusters blue
    colors = list(map(lambda x: '#b40426' if x == longest_index else '#3b4cc0', labels))
    plt.scatter(X[:,0], X[:,1], c=colors, marker="x", picker=True)
    plt.title('Clustered data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return(cluster_centres[longest_index])
    

clustering(0,0)