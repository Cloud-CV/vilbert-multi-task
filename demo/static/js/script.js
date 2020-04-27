window.onload = function() {
    var allLinks = document.querySelectorAll("div.new-tab a");
    for (var i = 0; i < allLinks.length; i++) {
        var currentLink = allLinks[i];
        currentLink.setAttribute("target", "_blank");
    }
}

function show_task_details(task_name, task_description) {
    $("#task-name").empty();
    var taskColDiv = $('<div class="col-md-12 fs-16" style="font-weight: 400;"></div>');
    var taskDescriptionRowDiv = $('<div class="row"><div class="col-md-12" style="font-weight: 300;">' + task_description + '</div></div>');

    $(taskColDiv).append(task_name);
    $(taskColDiv).append(taskDescriptionRowDiv);
    $("#task-name").append(taskColDiv);
}


// function get_task_data(task_id) {
//     var url = "https://vilbert.cloudcv.org/get_task_details/";
//     var url = url.concat(task_id)
//     var url = url.concat("/")
//     console.log(url);
//         $.ajax({
//             type: 'GET', // define the type of HTTP verb we want to use (GET)
//             url: url // the url where we want to GET
//         }).done(function(task_data) {
//             show_task_details(task_data.name, task_data.description)
//             // $(task_data.example).appendTo("#task-example");
//             var question =$("#question").val();
//             window.task_data = task_data;
//             $("task-example").text(task_data.example);
//             console.log(task_data);
//             if (question!="") {
//                 $("#question").val(question);
//             } else {
//                 $("#question").attr("placeholder", task_data.placeholder).val("").focus().blur();
//             }

//         });
// }