/*-----------------------------------------------------------------------------------

    Template Name: landio


    Note: This is Custom Js file

-----------------------------------------------------------------------------------

    [Table of contents]
    
    1. logodata
    2. clients-slider
    3. stickyHeader
    4. accordion-item
    5. Go to top
    6. loaded

-----------------------------------------------------------------------------------*/

/* 1. logodata */
jQuery(document).ready(function($){
    if ( $.isFunction($.fn.owlCarousel) ) {
    $('.quotation').owlCarousel({
            loop:true,
            dot:true,
            nav:false,
            autoplay:true,
            items:1,
            autoplayTimeout:3000,
            })

/* 2. clients-slider */
    $('.clients-slider').owlCarousel({
            loop:true,
            dot:false,
            nav:false,
            autoplay:true,
            autoplayTimeout:3000,
            responsive:{
                0:{
                    items:1
                },
                600:{
                    items:2
                },
                993:{
                    items:4
                },
                1200:{
                    items:5
                },
              }
            })
        }
});


/* 3. stickyHeader */
if ($("#stickyHeader")[0]){
 
    var new_scroll_position = 0;

        var last_scroll_position;

        var header = document.getElementById("stickyHeader");

        window.addEventListener('scroll', function(e) {

        last_scroll_position = window.scrollY;

        if (new_scroll_position < last_scroll_position && last_scroll_position > 100) {

          header.classList.remove("slideUp");

          header.classList.add("slideUp");
        } 

        else if (last_scroll_position < 100) {

          header.classList.remove("slideUp");

        } 

        else if (new_scroll_position > last_scroll_position) {

          header.classList.remove("slideUp");

          header.classList.add("slideUp");

        }

         new_scroll_position = last_scroll_position;

        });
      }



/* 4. accordion-item */
$('.accordion-item .heading').on('click', function(e) {
    e.preventDefault();

    if($(this).closest('.accordion-item').hasClass('active')) {
        $('.accordion-item').removeClass('active');
    } else {
        $('.accordion-item').removeClass('active');

        $(this).closest('.accordion-item').addClass('active');
    }
    var $content = $(this).next();
    $content.slideToggle(100);
    $('.accordion-item .content').not($content).slideUp('fast');
});

/* 5. Go to top */ 
function inVisible(element) {
  //Checking if the element is
  //visible in the viewport
  var WindowTop = $(window).scrollTop();
  var WindowBottom = WindowTop + $(window).height();
  var ElementTop = element.offset().top;
  var ElementBottom = ElementTop + element.height();
  //animating the element if it is
  //visible in the viewport
  if ((ElementBottom <= WindowBottom) && ElementTop >= WindowTop)
    animate(element);
}

function animate(element) {
  //Animating the element if not animated before
  if (!element.hasClass('ms-animated')) {
    var maxval = element.data('max');
    var html = element.html();
    element.addClass("ms-animated");
    $({
      countNum: element.html()
    }).animate({
      countNum: maxval
    }, {
      //duration 5 seconds
      duration: 5000,
      easing: 'linear',
      step: function() {
        element.html(Math.floor(this.countNum) + html);
      },
      complete: function() {
        element.html(this.countNum + html);
      }
    });
  }

}

//When the document is ready
$(function() {
  //This is triggered when the
  //user scrolls the page
  $(window).scroll(function() {
    //Checking if each items to animate are 
    //visible in the viewport
    $("h2[data-max]").each(function() {
      inVisible($(this));
    });
  })
});




let calcScrollValue = () => {
  let scrollProgress = document.getElementById("progress");
  let progressValue = document.getElementById("progress-value");
  let pos = document.documentElement.scrollTop;
  let calcHeight =
    document.documentElement.scrollHeight -
    document.documentElement.clientHeight;
  let scrollValue = Math.round((pos * 100) / calcHeight);
  if (pos > 100) {
    scrollProgress.style.display = "grid";
  } else {
    scrollProgress.style.display = "none";
  }
  scrollProgress.addEventListener("click", () => {
    document.documentElement.scrollTop = 0;
  });
  scrollProgress.style.background = `conic-gradient(#000 ${scrollValue}%, #ffffff00 ${scrollValue}%)`;
};

window.onscroll = calcScrollValue;
window.onload = calcScrollValue;

// 6. loaded


$(window).on('load', function () {
    $("body").addClass("page-loaded");
    ("loaded")
});