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

//securing the subscribe button against XSS
function secureForm(form){
  var form = document.getElementById("subscribeForm");
  var email = document.getElementById("email");
  var encoding = String(email.value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  email.value = encoding;
  //alert("email secured")
  form.innerHTML= "Email secured: " + email.value;
  form.submit();

}