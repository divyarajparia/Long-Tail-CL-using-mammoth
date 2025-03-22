def head_tail_segregate(trn_loader):
        # Step 1: Collect Boundary Points
        boundary_points = []
        for idx in range(len(trn_loader)):
            # Get the unique labels and their counts
            labels, counts = np.unique(trn_loader[idx].dataset.labels, return_counts=True)
            
            # Combine the labels and counts into a list of tuples and sort by counts
            label_counts = list(zip(labels, counts))
            label_counts_sorted = sorted(label_counts, key=lambda x: x[1], reverse=True)
            
            # Determine the index for the split
            split_index = len(label_counts_sorted) // 2
            
            # Get the boundary point m
            boundary_point_m = label_counts_sorted[split_index][1]
            boundary_points.append(boundary_point_m)
            
        print("Boundary Points (m) for all loaders:", boundary_points)

        # Step 2: Classify Classes Based on Boundary Points
        class_changes = {label: [] for label in range(100)}  # Assuming 50 classes from 0 to 49

        for idx in range(len(trn_loader)):
            # Get the unique labels and their counts
            labels, counts = np.unique(trn_loader[idx].dataset.labels, return_counts=True)
            
            # Combine the labels and counts into a list of tuples and sort by counts
            label_counts = list(zip(labels, counts))
            label_counts_sorted = sorted(label_counts, key=lambda x: x[1], reverse=True)
            
            # Get the boundary point for all loader
            for boundary_point_m in boundary_points:
                # Classify each class as 'head' or 'tail'
                for label, count in label_counts_sorted:
                    if count > boundary_point_m:
                        class_changes[label].append(0) # head
                    else:
                        class_changes[label].append(1) # tail

        # Print classification changes for each class
        for label, changes in class_changes.items():
            print(f"Class {label}: {changes}")
