// fazer uma função que pegar todos os botoões ao receber um click ele apaga os videos default e envia para o ajax qual é o id, ao mudar o src dos videos ele desabilita o id atual e ativa o anterior se existir
var dominio = ['Sul-Sudeste-Nordeste', 'Sudeste', 'Espírito Santo ', 'Região Metropolitana da Grande Vitória'];
$(document).ready(function () {

  function setVidNormal(vid) {
    $(vid).attr("width", 420)
  }

  $("#wind").click(function () {
    $('#actual').html('Velocidade do vento a 10 m de altura');
    $.ajax({
      url: "./assets/json/files.json",
      success: function (result) {
        var videos = result;
        for (let i = 0, j = 1; i < videos.videos.length; i++) {
          $("#vid" + j).attr("src", videos.videos[i].wind)
          setVidNormal("#vid" + j)
          j++
        }
      }
    });
  });

  $("#humidity").click(function () {
    $('#actual').html('Umidade específica na superfície');
    $.ajax({
      url: "assets/json/files.json", success: function (result) {
        var videos = result;
        for (let i = 0, j = 1; i < videos.videos.length; i++) {
          $("#vid" + j).attr("src", videos.videos[i].humidity)
          setVidNormal("#vid" + j)
          j++
        }
      }
    });
  });

  $("#temperature").click(function () {
    $('#actual').html('Temperatura do ar<br>e Pressão atmosférica na superfície');
    $.ajax({
      url: "assets/json/files.json", success: function (result) {
        var videos = result;
        for (let i = 0, j = 1; i < videos.videos.length; i++) {
          $("#vid" + j).attr("src", videos.videos[i].temperature)
          setVidNormal("#vid" + j)
          j++
        }
      }
    });
  });

  $("#radiation").click(function () {
    $('#actual').html('Radiação solar na superfície');
    $.ajax({
      url: "assets/json/files.json", success: function (result) {
        var videos = result;
        for (let i = 0, j = 1; i < videos.videos.length; i++) {
          $("#vid" + j).attr("src", videos.videos[i].radiation)
          setVidNormal("#vid" + j)
          j++
        }
      }
    });
  });

  $("#rain").click(function () {
    $('#actual').html('Preciptação na superfície');
    $.ajax({
      url: "assets/json/files.json", success: function (result) {
        var videos = result;
        for (let i = 0, j = 1; i < videos.videos.length; i++) {
          $("#vid" + j).attr("src", videos.videos[i].rain)
          setVidNormal("#vid" + j)
          j++
        }
      }
    });
  });
});