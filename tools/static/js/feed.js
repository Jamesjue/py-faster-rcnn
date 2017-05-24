/**
 * Created by suanmiao on 9/18/16.
 */
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
var csrftoken = getCookie('csrftoken');
function csrfSafeMethod(method) {
    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

$(document).ready(function () {
    $.ajaxSetup({
        beforeSend: function (xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });

    var base_url = location.protocol + "//" + location.host + "/";
    var default_avatar = "grumblr/static/img/avatar.jpg";
    var saved_list = [];
    var saved_cnt_text = 0;
    var saved_cnt_img = 0;
    $("#form-feed-post").submit(function (event) {
        event.preventDefault()
        var formData = new FormData();
        var input_content = $(".feed-post-input")[0].value;
        if (input_content.length == 0) {
            alert("Please input something before submit");
            return false;
        } else if (input_content.length > 42) {
            alert("Your input is too long");
            return false;
        }
        formData.append('content', input_content);
        var input_img = $(".feed-post-photo")[0].files[0];
        if (input_img != undefined) {
            if (!(input_img.name.toString().toLowerCase().endsWith(".png") ||
                input_img.name.toString().toLowerCase().endsWith(".jpg") ||
                input_img.name.toString().toLowerCase().endsWith(".jpeg"))) {
                return false;
                alert("The file you selected is not valid image !")
            }
            formData.append('img', input_img);
        }

        var success_callback = function (response) {
            var obj = JSON.parse(JSON.stringify(response));
            if (obj.status == 1) {
                alert("A new post is created ! ");
                $(".feed-post-input").val("");
                request_time_line()
            } else {
                alert(obj.msg);
            }
        };

        $.ajax({
            url: "/post",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });
    });

    var like_click = function (event) {
        var id = $(this).attr("id").toString().substr(5);
        var formData = new FormData();
        formData.append('post_id', id);
        formData.append('action', "like");

        var success_callback = function (response) {
            var obj = JSON.parse(JSON.stringify(response));
            if (obj.status == 1) {
                console.log("success like" + ("#like-num-" + id) + $("#like-num-" + id))
                $("#like-num-" + id).text(obj.likes);
            } else {
                alert(obj.msg);
            }
        };

        $.ajax({
            url: "/like_dislike",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });

    };


    var dislike_click = function (event) {
        var id = $(this).attr("id").toString().substr(8);
        var formData = new FormData();
        formData.append('post_id', id);
        formData.append('action', "dislike");

        var success_callback = function (response) {
            if (response.status == 1) {
                console.log("success dislike" + $(this).next(".feed-text-indicator"))
                console.log($("#dislike-num-" + id))
                $("#dislike-num-" + id).text(response.dislikes);
            } else {
                alert("dislike failure " + response.msg);
            }
        };

        $.ajax({
            url: "/like_dislike",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });

    };

    $(".logout").click(function (event) {
        var success_callback = function (response) {
            location.href = "/login"
        };

        $.ajax({
            url: "/logout",
            type: "POST",
            async: true,
            cache: false,
            success: success_callback,
        });

    })

    $(".profile-follow").click(function (event) {
        var target_id = $(this).attr("id").toString();
        var formData = new FormData();
        formData.append('target_id', target_id);

        console.log($(this).text())
        if ($(this).text() === "Follow") {
            formData.append('action', "follow");
        } else {
            formData.append('action', "unfollow");
        }

        var success_callback = function (response) {
            if (response.status == 1) {
                if ($(".profile-follow").text() === "Follow") {
                    $(".profile-follow").text("Unfollow")
                } else {
                    $(".profile-follow").text("Follow")
                }
                $(".text-follower-following").text(" Follower: " + response.follower + "  Following: " + response.following)
                console.log("success follow")
            } else {
                alert("follow failure " + response.msg);
            }
        };

        $.ajax({
            url: "/follow_unfollow",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });
    })

    var comment_submit = function (event) {
        event.preventDefault();
        var formNode = $(this)[0];
        var formData = new FormData();
        var pid = $(this).data("pid");
        var inputNode = $(formNode).find(".feed-comment-input");
        var content = inputNode.val();
        console.log("pid " + pid)
        console.log("content " + content)

        formData.append('pid', pid);
        formData.append('content', content);

        var success_callback = function (response) {
            if (response.status == 1) {
                console.log("success comment" + JSON.stringify(response))

                //show comments
                var list = $("#comment-list-" + pid);
                load_comment(pid);
                list.show();

                $("#trigger-" + pid).text("Hide Comment");
                $("#comment-num-" + pid).text(response.comments + "");
                inputNode.val("");
            } else {
                alert("comment failure " + response.msg);
            }
        };

        $.ajax({
            url: "/comment",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });
    }

    var list_item_text = $(".feed-item-template-text");
    var list_item_img = $(".feed-item-template-img");

    var comment_item_template = $(".feed-comment-item-template")

    var trigger_comment_visibility = function () {
        var pid = $(this).data("pid");
        var list = $("#comment-list-" + pid);
        var isVisible = list.is(":visible");
        if (isVisible) {
            $(this).text("Show Comment");
            list.hide();
        } else {
            load_comment(pid);
            $(this).text("Hide Comment");
            list.show();
        }
    }

    var bind_comment_data = function (data, pid) {
        var item = comment_item_template.clone();
        item.removeClass("feed-comment-item-template");
        item.find(".feed-comment-avatar-container").attr("href", base_url + "user/" + data.user.id);
        if (data.user.avatar) {
            item.find(".feed-comment-avatar").attr("src", base_url + data.user.avatar);
        } else {
            item.find(".feed-comment-avatar").attr("src", base_url + default_avatar);
        }
        item.find(".feed-comment-content").text(data.content)
        item.find(".feed-comment-time").text(data.date)

        var list = $("#comment-list-" + pid);
        list.append(item)
    }

    var load_comment = function (pid) {
        console.log("load comment for " + pid)
        var success_callback = function (response) {
            if (response.status == 1) {
                var list = response.data.list;
                $("#comment-list-" + pid).empty();
                for (var i = 0; i < list.length; i++) {
                    var p = list[i];
                    bind_comment_data(p, pid);
                }
            } else {
                alert("request comments failure " + response.data.msg);
            }
        };
        var formData = new FormData();
        formData.append("pid", pid);
        $.ajax({
            url: "/comments",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });

    }


    var bind_feed_data = function (data) {
        var item;
        if (data.type === "IMG") {
            item = list_item_img.clone();
            item.removeClass("feed-item-template-img");
            item.addClass("feed-item-" + data.id);
            item.find(".feed-img-container").attr("href", base_url + data.img);
            item.find(".feed-img-href").attr("href", base_url + data.img);
            item.find(".feed-img-content").attr("src", base_url + data.img);

        } else {
            item = list_item_text.clone();
            item.removeClass("feed-item-template-text");
        }
        item.find(".feed-avatar-container").attr("href", base_url + "user/" + data.user.id);
        if (data.user.avatar) {
            item.find(".feed-avatar").attr("src", base_url + data.user.avatar);
        } else {
            item.find(".feed-avatar").attr("src", base_url + default_avatar);
        }

        item.find(".feed-author").attr("href", base_url + "user/" + data.user.id);
        item.find(".feed-author").text(data.user.first_name + data.user.last_name);
        item.find(".feed-time").text(data.date);
        item.find(".feed-content-text").text(data.content);
        item.find(".feed-img-like").attr("id", "like-" + data.id);
        item.find(".feed-img-dislike").attr("id", "dislike-" + data.id);

        item.find(".feed-img-like").click(like_click);
        item.find(".feed-img-dislike").click(dislike_click);

        item.find(".feed-text-indicator-like").attr("id", "like-num-" + data.id);
        item.find(".feed-text-indicator-dislike").attr("id", "dislike-num-" + data.id);

        item.find(".feed-comment-list").attr("id", "comment-list-" + data.id);
        item.find(".feed-comment-list").hide();
        item.find(".feed-comment-form").attr("id", "comment-form-" + data.id);
        item.find(".feed-comment-form").data("pid", data.id);
        item.find(".feed-comment-form").submit(comment_submit);

        item.find(".feed-comment-trigger").data("pid", data.id);
        item.find(".feed-comment-trigger").click(trigger_comment_visibility);
        item.find(".feed-comment-trigger").attr("id", "trigger-" + data.id);

        item.find(".feed-text-indicator-like").text(" " + data.likes);
        item.find(".feed-text-indicator-dislike").text(" " + data.dislikes);
        item.find(".feed-text-indicator-comment").text(" " + data.comments);
        item.find(".feed-text-indicator-comment").attr("id", "comment-num-" + data.id);

        $("#feed-list").prepend(item)
    };

    var get_latest_post_time = function () {
        if (saved_list.length == 0) {
            return 0;
        } else {
            return saved_list[0].time;
        }
    }

    var request_time_line = function () {
        var type = $("#page-type").val();

        var success_callback = function (response) {
            if (response.status == 1) {
                var post_list = response.data.post.list;
                for (var i = 0; i < post_list.length; i++) {
                    var p = post_list[i];
                    bind_feed_data(p);
                    saved_list.unshift(p);
                }
                $(".feed-img-like").on("click", like_click);
                $(".feed-img-dislike").on("click", dislike_click);
                saved_cnt_img += response.data.post.img;
                saved_cnt_text += response.data.post.text;
                $("#feed-filter-count-all").text((saved_cnt_img + saved_cnt_text) + "")
                $("#feed-filter-count-text").text(saved_cnt_text + "")
                $("#feed-filter-count-img").text(saved_cnt_img + "")
            } else {
                alert("request timeline failure " + response.data.msg);
            }
        };
        var formData = new FormData();
        if (type.startsWith("user")) {
            formData.append("type", "user");
            var uid = type.split("-")[1];
            formData.append("uid", uid);
        } else if (type === "follower_stream") {
            formData.append("type", "follower_stream");
        } else if (type === "global") {
            formData.append("type", "global");
        }
        formData.append("after", get_latest_post_time());
        $.ajax({
            url: "/timeline",
            type: "POST",
            data: formData,
            contentType: false,
            processData: false,
            async: true,
            cache: false,
            success: success_callback,
        });
    }
    request_time_line();
    window.setInterval(request_time_line, 5000);
    return false;
})
