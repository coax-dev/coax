// (function() {
//   // Makes "Other Method" admonitions collapsible. See also custom.css

//   var admonitions = document.getElementsByClassName("admonition-other-methods");

//   for (var i = 0; i < admonitions.length; i++) {
//     admonition = admonitions[i];
//     admonition.insertAdjacentText(
//       "beforeEnd",
//       "(these are methods that you typically won't need to access yourself)")

//     var title = admonition.getElementsByClassName("admonition-title")[0];

//     title.addEventListener("click", function() {

//       this.classList.toggle("active");
//       methods = this.parentElement.getElementsByClassName("method");

//       for (var j = 0; j < methods.length; j++) {

//         var method = methods[j];
//         if (method.style.display === "block") {
//           method.style.display = "none";
//         } else {
//           method.style.display = "block";
//         }

//       }

//     });

//   }

// })()
