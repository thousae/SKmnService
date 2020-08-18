let frame = null;
const timeout = 100;

window.onload = () => {
    // This code will be executed when window has been loaded
    $('.ui.form')
        .form({
            fields: {
                url: 'url'
            }
        })
    ;

    $('section#section-2 .ui.segment')
        .popup()
    ;

    frame = $('iframe');
}

window.onresize = () => {
    resizeFrame();
}

function resizeFrame() {
    if (frame) {
        frame.attr('height', frame.width() * 9 / 16);
    }
}

function showModal(id) {
    $(`.ui.modal`)
        .modal({
            onApprove   : () => false,
            onHidden    : () => {
                frame[0].contentWindow
                    .postMessage(
                        '{"event":"command","func":"' + 'stopVideo' + '","args":""}',
                        '*'
                    )
                ;
            }
        })
        .modal('show')
    ;
    initContents(id);
    activeContent(id, 1);
}

function getStepByIdAndIndex(id, index) {
    if (index === 1) {
        return $(`.ui.modal #${id}-step-${index}`);
    } else {
        return $(`.ui.modal #step-${index}`);
    }
}

function getSectionByIdAndIndex(id, index) {
    if (index === 1) {
        return $(`.ui.modal #${id}-section-${index}`);
    } else {
        return $(`.ui.modal #section-${index}`);
    }
}

function initContents(id) {
    $(`.ui.modal div.header`).hide();
    $('form.ui.form .ui.error.message').hide();
    $('iframe').hide();
    $('#description').show();
    $('.ui.modal .ui.segment').show();
    $(`div.header#${id}-header`).show();

    $('.ui.modal .step').removeClass('active').removeClass('completed').hide();
    $('.ui.modal section').hide();

    for (let i = 1; i <= 3; i++) {
        const target = getStepByIdAndIndex(id, i);
        target.show();
    }
}

function isUrl(text) {
    const reUrl = /[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)/ig;
    return reUrl.test(text);
}

function urlToEmbed(url) {
    const youtube = /^(?:https?:\/\/)?(?:www\.)?(?:(?:youtube.com)|(?:youtu.be)).*(?:[\/?=](?<code>[a-zA-z0-9]{11})[\/?=]?).*$/;
    if (youtube.test(url)) {
        return url.replace(youtube, 'https://www.youtube.com/embed/$1');
    } else {
        return url;
    }
}

function getUrl() {
    return $('form.ui.form input#url').val();
}

function getUserID() {
    return new Promise(resolve => {
        $.ajax('/get-uid', {
            type    : 'post',
            success : data => resolve(data)
        });
    });
}

async function getResult() {
    const userID = await getUserID();
    return new Promise(resolve => {
        const intervalID = setInterval(() => {

        }, timeout);
    });
}

function getNextButtonEvent(id, index) {
    switch (index) {
        case 1:
            return async () => {
                const url = getUrl();
                if (id === 'upload' || isUrl(url)) {
                    const button = $('.positive.button');
                    button
                        .addClass('disabled')
                        .addClass('loading')
                    ;

                    const result = await getResult();

                    button
                        .removeClass('disabled')
                        .removeClass('loading')
                    ;
                    activeContent(id, index + 1, result);
                } else {
                    $('form.ui.form .ui.error.message').show();
                }
            };

        case 2:
            return () => activeContent(id, index + 1);

        case 3:
            return () => $('.ui.modal').modal('toggle');
    }
}

function activeContent(id, step, result = '') {
    for (let i = 1; i <= 3; i++) {
        const target = getStepByIdAndIndex(id, i);
        const section = getSectionByIdAndIndex(id, i);
        if (i < step) {
            section.hide();
            target.removeClass('active').addClass('completed');
        } else if (i === step) {
            section.show();
            target.addClass('active');
        }
    }

    if (result) {
        $('#result').text(result);
    }

    const button = $(`.ui.modal .actions .positive.button`);
    button.unbind();
    button.click(getNextButtonEvent(id, step));
}

function showVideo() {
    const url = getUrl();
    const embed = urlToEmbed(url);

    frame
        .attr('src', embed)
        .show()
    ;
    resizeFrame();

    $('.ui.modal .ui.segment').hide();
    hideDescription();
}

function hideDescription() {
    $('#description').hide();
}