// fazer uma função que pegar todos os botoões ao receber um click ele apaga os videos default e envia para o ajax qual é o id, ao mudar o src dos videos ele desabilita o id atual e ativa o anterior se existir
var dominio = ['Estado da Bahia', 'Regi&atilde;o Metropolitana', 'Salvador'];
$(document).ready(function(){
  
  $("#wind").click(function(){
    $('#actual').html('Velocidade do vento');
    $("#videos").html(' ');
    $.ajax({url: "assets/json/files.json", success: function(result){
      $.each(result, function(i, field){
          for(k in field) {
            var j = k;
            j++;
            $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
            '" src="' + field[k].wind + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
          }
        }); 
      }});
    });

      $("#humidity").click(function(){
        $('#actual').html('Umidade especifica');
        $("#videos").html(' ');
        $.ajax({url: "assets/json/files.json", success: function(result){
          $.each(result, function(i, field){
            for(k in field) {
              var j = k;
              j++;
              $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
              '" src="' + field[k].humidity + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
            }
          }); 
        }});
      });
     
      $("#temperature").click(function(){
        $('#actual').html('Temperatura do ar<br>e Pressao atmosferica');
        $("#videos").html(' ');
        $.ajax({url: "assets/json/files.json", success: function(result){
          $.each(result, function(i, field){
            for(k in field) {
              var j = k;
              j++;
              $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
              '" src="' + field[k].temperature + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
            }
          }); 
        }});
      });

      $("#radiation").click(function(){
        $('#actual').html('Radiacao solar');
        $("#videos").html(' ');
        $.ajax({url: "assets/json/files.json", success: function(result){
          $.each(result, function(i, field){
            for(k in field) {
              var j = k;
              j++;
              $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
              '" src="' + field[k].radiation + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
            }
          }); 
        }});
      });

      $("#rain").click(function(){
        $('#actual').html('Preciptacao');
        $("#videos").html(' ');
        $.ajax({url: "assets/json/files.json", success: function(result){
          $.each(result, function(i, field){
            for(k in field) {
              var j = k;
              j++;
              $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
              '" src="' + field[k].rain + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
            }
          }); 
        }});
      });

      $("#latentHeat").click(function(){
        $('#actual').html('Calor latente');
        $("#videos").html(' ');
        $.ajax({url: "assets/json/files.json", success: function(result){
          $.each(result, function(i, field){
            for(k in field) {
              var j = k;
              j++;
              $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
              '" src="' + field[k].latentHeat + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
            }
          }); 
        }});
      });

      $("#sensitiveHeatFlow").click(function(){
        $('#actual').html('Fluxo de calor sensivel');
        $("#videos").html(' ');
        $.ajax({url: "assets/json/files.json", success: function(result){
          $.each(result, function(i, field){
            for(k in field) {
              var j = k;
              j++;
              $("#videos").append('<div class="col text-center"><h6 class="text-lightdark mb-3">DOM&Iacute;NIO ' + j + ' - ' + dominio[j-1] + '</h6><video id="vid" autoplay loop="true" controls width="320" height="264"><source id="video' + k + 
              '" src="' + field[k].sensitiveHeatFlow + '" />To view this video please enable JavaScript, and consider upgrading to a web browser that <a href="html5-video-support.html" target="_blank">supports HTML5 video</a></video></div>'); 
            }
          }); 
        }});
      });
});