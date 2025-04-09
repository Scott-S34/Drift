// Get the button element
var bookButton = document.getElementById("bookButton");

// Get the message element
var message = document.getElementById("message");

// Add event listener to the button
bookButton.addEventListener("click", function() {
  // Show the message
  message.style.display = "block";
});
// Get the button element
var bookTableButton = document.getElementById("bookTableButton");

// Get the form section
var formSection = document.getElementById("formSection");

// Add event listener to the button
bookTableButton.addEventListener("click", function() {
  // Show the form section
  formSection.style.display = "block";
  
  // Scroll to the form section
  formSection.scrollIntoView({ behavior: 'smooth' });
});


document.getElementById("bookTableButton").addEventListener("click", function() {
    window.location.href = "booktable.html";
});
document.getElementById("submitButton").addEventListener("click", function() {
    window.location.href = "thankyou.html";
});

//securing the subscribe button against Cross-Site Scripting XSS
function secureForm(){
  //get the form with the email subscription
  var form = document.getElementById("subscribeForm");
  //get the email input box
  var email = document.getElementById("email");
  //replace some special characters with their URL encoding 
  var encoding = String(email.value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  //set the value of the email to the URL encoding version of the user input
  email.value = encoding;
  //alert("email secured")
  //return the email to check that XSS has been mitigated
  form.innerHTML= "Email secured: " + email.value;
  //submit the subscribe form
}

//providing a security mechanism against file upload vulnerabilities
//these help to prevent big files or malicious code files from being executed and destroying the website
function secureFileUploading(){
  var scanFunction = document.getElementById("scanner");
  //get the file uploaded by the user
  var fileUpload = document.getElementById('upload');
  //acquire the file size
  var fileSize = fileUpload.files[0].size;
  //set a limit for the maximum file size
  var maxSize = 12  * 1024; //12KB;
  //check the size of the file
  alert(fileSize + "B" + ", " + (fileSize /1024) + "MB");
  //check that the filesize is in the limit
  if(fileSize > maxSize){
    alert("Sorry! File size is too big!");
    return;
  }

  //acquire the supplied filename and its extension
  var fileName = fileUpload.value;
  var fileParts = fileName.split(".");
  var extension = String(fileParts.pop().toLowerCase());
  alert("Extension = " + extension);  
  //if the file type is jpg, jpeg, or png, proceed 
  if( fileSize < maxSize && ( (extension === "jpg") || (extension === "png") || (extension == "jpeg") ) ){
    alert("Thank you! Result is loading");
    scanFunction.submit();
  } else{   
    alert("Insufficient file extension, image should be either png, jpg or jpeg");
    return;
  }
}