 
function indexImages() {
    var loader = document.getElementById("loader");
    var button = document.getElementById("index-button");
    var progress = document.getElementById("progress");
    
    // Hide the index button and display the loader
    button.style.display = "none";
    loader.style.display = "flex";
    progress.innerHTML = "Indexing in progress...";
    
    // Start the indexing process by making an AJAX request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/index", true);
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            // Hide the loader
            loader.style.display = "none";
            
            if (xhr.status === 200) {
                try {
                    var response = JSON.parse(xhr.responseText);
                    progress.innerHTML = "✅ " + response.message;
                    
                    // Show success message and reset button after 3 seconds
                    setTimeout(function() {
                        button.style.display = "block";
                        progress.innerHTML = "Ready to index images";
                    }, 3000);
                    
                } catch (e) {
                    progress.innerHTML = "✅ Indexing completed!";
                    setTimeout(function() {
                        button.style.display = "block";
                        progress.innerHTML = "Ready to index images";
                    }, 3000);
                }
            } else {
                progress.innerHTML = "❌ Error occurred during indexing";
                button.style.display = "block";
            }
        }
    };
    
    xhr.onerror = function() {
        loader.style.display = "none";
        progress.innerHTML = "❌ Network error occurred";
        button.style.display = "block";
    };
    
    xhr.send();
}

// Auto-update progress for long-running operations
function updateProgress() {
    var progressTexts = [
        "Analyzing images...",
        "Extracting features...",
        "Computing descriptors...",
        "Saving to index..."
    ];
    
    var index = 0;
    var interval = setInterval(function() {
        var progress = document.getElementById("progress");
        if (progress && progress.innerHTML.includes("progress")) {
            progress.innerHTML = progressTexts[index % progressTexts.length];
            index++;
        } else {
            clearInterval(interval);
        }
    }, 2000);
}
