var text_;

$(document).ready(function(){
  // fetch all DOM elements for the input
  text_ = document.getElementById("text");

})

$(document).on('click','#submit',function(){
    // on clicking submit fetch values from DOM elements and use them to make request to our flask API
    var text = text_.value;

    if(text == ""){
      // you may allow it as per your model needs
      // you may mark some fields with * (star) and make sure they aren't empty here
      alert("empty fields not allowed");
      console.log("entered")
    }
    else{
      console.log("entered")
      // replace <username> with your pythonanywhere username
      // also make sure to make changes in the url as per your flask API argument names
      // var requestURL = "https://<username>.pythonanywhere.com/?text="+text;
      var requestURL = "http://localhost:5000/classify/"+text;
      console.log(requestURL); // log the requestURL for troubleshooting
      $.getJSON(requestURL, function(data) {
        console.log(data); // log the data for troubleshooting
        prediction = data['json_key_for_the_prediction'];
      });
      // following lines consist of action that would be taken after the request has been read
      // for now i am just changing a <h2> tag's inner html using jquery
      // you may simple do: alert(prediction);
      $(".result").html("Prediction is:" + prediction);
    }
  });
