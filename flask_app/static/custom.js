function submit_message(message) {
        $.post( "/send_message", {message: message}, handle_response);

        function handle_response(data) {
          // append the bot repsonse to the div
          var response=JSON.stringify(data);
          // alert(response);
            var today = new Date();
            var date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
            var time = today.getHours()+':'+today.getMinutes()+':'+today.getSeconds();
            var datetime = date + ' ' + time
          $('.chat-container').append(`
                <div class="chat-message col-md-5 bot-message">
                    <p>${datetime} - ChatBot:</p>
                    <p>${data.message}<p>
                </div>
          `)
          // remove the loading indicator
          $( "#loading" ).remove();
          $(".chat-container").scrollTop($(".chat-container")[0].scrollHeight);
          //$(window).scrollTop($('#input_message').position().top);
        }
    }

$('#target').on('submit', function(e){
        e.preventDefault();
        var today = new Date();
        var date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
        var time = today.getHours()+':'+today.getMinutes()+':'+today.getSeconds();
        var datetime = date + ' ' + time
        const input_message = $('#input_message').val()
        // return if the user does not enter any text
        if (!input_message) {
          return
        }

        $('.chat-container').append(`
            <div class="chat-message offset-md-7 human-message">
                <p>${datetime} - User: </p>
                <p>${input_message}<p>
            </div>
        `)

        // loading
        $('.chat-container').append(`
            <div class="chat-message text-center col-md-2 bot-message" id="loading">
                <b>...</b>
            </div>
        `)

        // clear the text input
        $('#input_message').val('')

        // send the message
        submit_message(input_message)
    });

window.onload = function() {
    var today = new Date();
    var date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
    var time = today.getHours()+':'+today.getMinutes()+':'+today.getSeconds();
    document.getElementById("datenow").innerHTML = date + ' '+ time + ' - ChatBot:';
};