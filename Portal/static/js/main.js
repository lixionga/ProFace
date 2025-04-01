/*===================================================
Project:   Yufa 
Auther: Mohamed Amin
Version:    1.0
Created:   2 Nov, 2020
Last Update: 10 Aug, 2023
Template Description:    multipurpose HTML5 Template 
====================================================*/

//GLOBAL VARIBALES
var //selector vars
  main_window = $(window),
  root = $("html, body"),
  bdyOnePage = $("body.landing-page-demo "),
  pageHeader = $("#page-header"),
  navMain = $("nav.main-navbar"),
  navMenuWraper = $(".navbar-menu-wraper"),
  hasSubMenu = $(".has-sub-menu"),
  onePage_navLink = $(
    ".landing-page-demo .main-navbar .nav-link, .landing-page-demo .goto-link"
  ),
  pageHero = $("#page-hero"),
  backToTopButton = $(".back-to-top"),
  heroVegasSlider = $(".hero-vegas-slider"),
  heroSwiperSlider = ".hero-swiper-slider .swiper-container",
  textInput = $("form.main-form .text-input"),
  tabLink = $(".ma-tabs .tabs-links .tab-link"),
  togglerLink = $(".ma-tabs .toggler-links .toggler"),
  portfolioGroup = $(".portfolio .portfolio-group"),
  // Measurements vars
  navMainHeight = navMain.innerHeight(),
  pageHeroHeight = pageHero.innerHeight(),
  //class Names Strings vars

  hdrStandOut = "header-stand-out",
  inputHasText = "has-text",
  // condetionals vars
  counterShowsUp = false;

$(function () {
  ("use strict");
  /*-----------------  START GENERAL FUNCTIONS  -----------------*/

  // function to place a line under the active link on the tabs components
  function adjust_tabLink_B_line() {
    // to Move the tab-link bottom line
    if ($(".ma-tabs .tabs-links-list").length) {
      var pageDir = $("body").css("direction");

      var $line = tabLink.parent(".tabs-links-list").find(".b-line");
      var activTabLink = tabLink.parent(".tabs-links-list").children(".active");
      var eleParentWidth = activTabLink.parent(".tabs-links-list").innerWidth();

      var eleWidth = activTabLink.innerWidth();
      var eleLeft = activTabLink.position().left;
      var eleRight = eleParentWidth - (eleLeft + eleWidth);

      if (pageDir === "ltr") {
        $line.css({
          left: eleLeft + "px",
          width: eleWidth + "px",
          // "max-width": eleWidth - 10 + 'px',
        });
      } else
        $line.css({
          right: eleRight + "px",
          width: eleWidth + "px",
          // "max-width": eleWidth - 10 + 'px',
        });
    }
  }

  // function to fire the conter plugin
  function fireCounter() {
    if ($(".stats-counter").length) {
      if (jQuery().countTo && counterShowsUp === false) {
        var pos = $(".stats-counter").offset().top;

        if (main_window.scrollTop() + main_window.innerHeight() - 50 >= pos) {
          $(".counter").countTo();
          counterShowsUp = true;
        }
      }
    }
  }
  // // function to fire the conter plugin
  // function fireCounter() {
  //     if ($('.stats-counter').length) {
  //         if (jQuery().countTo && counterShowsUp === false) {
  //             // if ($('.counter').length && counterShowsUp === false) {
  //             var pos = $('.stats-counter').position();

  //             if (((main_window.scrollTop() + main_window.innerHeight()) >= pos.top)) {
  //                 $('.counter').countTo();
  //                 counterShowsUp = true;
  //             }
  //             // }

  //         }
  //     }
  // }

  /*-----------------  END GENERAL FUNCTIONS  -----------------*/

  /*----------------- Start Calling global function -----------------*/

  /*this functions must fires on the page ready to load*/

  //to adjust tabs links  underline
  adjust_tabLink_B_line();

  // to fire the counter when its section appear on screen
  fireCounter();

  /*----------------- End Calling global function -----------------*/

  // fix to make sure vegas slider allways taking the full height of the hero section
  if (heroVegasSlider.length) {
    // only on pages that has vegas slider

    if (pageHeroHeight < $(this).innerHeight()) {
      $(pageHero).css("height", "100vh");
    }
  }

  // to remove class from navbar if the page refreshed and the scrolltop of the window > 50px
  if (main_window.scrollTop() > 100) {
    $(pageHeader).toggleClass(hdrStandOut);
    $(backToTopButton).toggleClass("show");
  }

  if ($(textInput).length) {
    if ($(textInput).val().trim() !== "")
      $(textInput).parent().addClass(inputHasText);
    else $(textInput).parent().removeClass(inputHasText);

    //check if the form input has data or not while focusing out
    //from the input to set the label
    //in the right place by the css rules.
    $(textInput).on("focusout", function () {
      if ($(this).val().trim() !== "") $(this).parent().addClass(inputHasText);
      else $(this).parent().removeClass(inputHasText);
    });
  }

  /* ----------------- End page loading Actions * ----------------- */

  /* ----------------- Start onClick Actions * ----------------- */

  //  Start Smooth Scrolling To page Sections
  $(onePage_navLink).on("click", function (e) {
    var link = $(this).attr("href");
    var currentMainNavHeight = navMain.innerHeight();

    if (link.charAt(0) === "#") {
      e.preventDefault();
      var target = this.hash;
      $(root).animate(
        {
          scrollTop: $(target).offset().top - currentMainNavHeight + 1,
        },
        500
      );
    }
  });

  //End Smooth Scrolling To page Sections

  $(".navbar-nav").on("click", function (e) {
    e.stopPropagation();
  });

  //  open and close menu btn
  $(".menu-toggler-btn, .navbar-menu-wraper ").on("click", function () {
    $(".menu-toggler-btn").toggleClass("close-menu-btn");
    navMenuWraper.toggleClass("show-menu");

    //  add/remove  .header-stand-out  class to .main-navbar when menu-toogler-clicked

    //  if the menu is opened
    if ($(".show-menu").length) {
      // add .header-stand-out class to .main-nav
      if (!pageHeader.hasClass(hdrStandOut))
        $(pageHeader).addClass(hdrStandOut);
    } else {
      // remove .header-stand-out class to .main-nav in case the window scrolltop less than 50px
      if (
        pageHeader.hasClass(hdrStandOut) &&
        main_window.scrollTop() < 50 &&
        main_window.innerWidth > "991px"
      )
        $(pageHeader).removeClass(hdrStandOut);
    }
  });

  //showing navbar sub-menus
  hasSubMenu.on("click", function (e) {
    e.stopPropagation();
    if (!(main_window.innerWidth() > 1199)) {
      $(this).children(".sub-menu").slideToggle();
    }
  });

  // Start Smooth Scrolling To Window Top When Clicking on Back To Top Button
  $(backToTopButton).on("click", function () {
    root.animate(
      {
        scrollTop: 0,
      },
      1000
    );
  });
  // End Smooth Scrolling To Window Top When Clicking on Back To Top Button

  // Start tabs navigation

  // Start Regular Tabs
  $(tabLink).on("click", function () {
    var target = $(this).attr("data-target");

    $(this).addClass("active").siblings().removeClass("active");

    $(target)
      .addClass("visibale-tab")
      .siblings(".tab-content")
      .removeClass("visibale-tab");

    adjust_tabLink_B_line();
  });
  //End Regular  Tabs

  // Start  Switch  Toggler Tabs

  //When click on the left Switch btn
  $(".switch-left ").on("click", function () {
    //make sure the toggler checkbox unchecked
    if ($(".toggle-btn").prop("checked") === true) {
      $(".toggle-btn").prop("checked", false);
    }

    // 1-) add .active class to the pressed btn & remove it from sibilings
    $(this).addClass("active").siblings().removeClass("active");

    // 2-) show the wanted tab
    var target = $(this).attr("data-target");
    var currentSection = $(this).parents(".ma-tabs");
    $(currentSection)
      .find(target)
      .addClass("visibale-tab")
      .siblings(".tab-content")
      .removeClass("visibale-tab");
  });

  $(".switch-right").on("click", function () {
    //make sure the toggler checkbox  checked
    if ($(".toggle-btn").prop("checked") === false) {
      $(".toggle-btn").prop("checked", true);
    }

    // 1-) add .active class to the pressed btn & remove it from sibilings
    $(this).addClass("active").siblings().removeClass("active");

    // 2-) show the wanted tab
    var target = $(this).attr("data-target");
    var currentSection = $(this).parents(".ma-tabs");
    $(currentSection)
      .find(target)
      .addClass("visibale-tab")
      .siblings(".tab-content")
      .removeClass("visibale-tab");
  });

  // Do the same as the clicked switch-btns but when press on the checkbox toggler it self
  $(".toggle-btn").on("click", function () {
    if ($(this).prop("checked")) {
      // 1-) add .active class to the pressed btn & remove it from sibilings
      $(this)
        .parent()
        .siblings(".switch-left")
        .removeClass("active")
        .siblings(".switch-right")
        .addClass("active");

      // 2-) show the wanted tab
      var target = $(this)
        .parent()
        .siblings(".switch-right")
        .attr("data-target");
      var currentSection = $(this).parents(".ma-tabs");
      $(currentSection)
        .find(target)
        .addClass("visibale-tab")
        .siblings(".tab-content")
        .removeClass("visibale-tab");
    } else {
      // 1-) add .active class to the pressed btn & remove it from sibilings
      $(this)
        .parent()
        .siblings(".switch-left")
        .addClass("active")
        .siblings(".switch-right")
        .removeClass("active");

      // 2-) show the wanted tab
      var target = $(this)
        .parent()
        .siblings(".switch-left")
        .attr("data-target");
      var currentSection = $(this).parents(".ma-tabs");
      $(currentSection)
        .find(target)
        .addClass("visibale-tab")
        .siblings(".tab-content")
        .removeClass("visibale-tab");
    }
  });

  // End Switch Toggler Tabs

  //End tabs navigation
  /* ----------------- End onClick Actions ----------------- */

  /* ----------------- Start onScroll Actions ----------------- */

  main_window.on("scroll", function () {
    if ($(this).scrollTop() > 50) {
      //show back to top btn
      backToTopButton.addClass("show");
    } else {
      //hide back to top btn
      backToTopButton.removeClass("show");
    }

    // to add/remove a class that makes navbar stands out
    // by changing its colors to the opposit colors
    if ($(this).innerWidth() > 991) {
      if ($(this).scrollTop() > 50) {
        if (!$(pageHeader).hasClass(hdrStandOut))
          $(pageHeader).addClass(hdrStandOut);
      } else {
        if ($(pageHeader).hasClass(hdrStandOut))
          $(pageHeader).removeClass(hdrStandOut);
      }
    } else {
      // on screens smaller than 992px always add header-standout class
      if (!$(pageHeader).hasClass(hdrStandOut)) {
        $(pageHeader).addClass(hdrStandOut);
      }
    }

    // to make sure the counter will start counting whit its section apear on the screen
    fireCounter();
  });

  /* ----------------- End onScroll Actions ----------------- */

  /*************Start Contact Form Functionality************/

  const contactForm = $("#contact-us-form"),
    userName = $("#user-name"),
    userEmail = $("#user-email"),
    msgSubject = $("#msg-subject"),
    msgText = $("#msg-text"),
    submitBtn = $("#submit-btn");

  let isValidInput = false,
    isValidEmail = false;

  function ValidateNotEmptyInput(input, errMsg) {
    if (input.length) {
      if (input.val().trim() === "") {
        $(input).siblings(".error-msg").text(errMsg).css("display", "block");
        isValidInput = false;
      } else {
        $(input).siblings(".error-msg").text("").css("display", "none");
        isValidInput = true;
      }
    }
  }

  function validateEmailInput(emailInput) {
    let pattern =
      /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;

    if (pattern.test(emailInput.val()) === false) {
      $(emailInput)
        .siblings(".error-msg")
        .text("Please Enter a valid Email")
        .css("display", "block");
      isValidEmail = false;
    } else {
      $(emailInput).siblings(".error-msg").text("").css("display", "none");
      isValidEmail = true;
    }
  }

  submitBtn.on("click", function (e) {
    e.preventDefault();

    ValidateNotEmptyInput(userName, "Please Enter Your Name");
    ValidateNotEmptyInput(userEmail, "Please Enter Your Email");
    ValidateNotEmptyInput(msgSubject, "Please Enter Your subject");
    ValidateNotEmptyInput(msgText, "Please Enter Your Message");
    validateEmailInput(userEmail);

    if (isValidInput && isValidEmail) {
      $.ajax({
        type: "POST",
        url: contactForm.attr("action"),
        data: contactForm.serialize(),

        success: function (data) {
          $(".done-msg")
            .text("Thank you, Your Message Was Received!")
            .toggleClass("show");
          setTimeout(function () {
            $(".done-msg").text("").toggleClass("show");
          }, 7500);
          contactForm[0].reset();
        },
      });
      return false;
    }
  });

  /*************End Contact Form Functionality************/

  /* ----------------- Start Window Resize Actions ----------------- */

  main_window.on("resize", function () {
    // a fix to make sure vigas slider always taking the full height of the hero section
    if (heroVegasSlider.length) {
      if (pageHeroHeight < $(this).innerHeight()) {
        $(pageHero).css("height", "100vh");
      }
    }

    if (main_window.innerWidth() > 991) {
      if (navMenuWraper.hasClass("show-menu")) {
        navMenuWraper.removeClass("show-menu");
        $(".menu-toggler-btn").toggleClass("close-menu-btn");
      }

      if (hasSubMenu.children(".sub-menu").css("display", "none")) {
        hasSubMenu.children(".sub-menu").css("display", "block");
      }
    } else {
      if (hasSubMenu.children(".sub-menu").css("display", "block")) {
        hasSubMenu.children(".sub-menu").css("display", "none");
      }
    }

    adjust_tabLink_B_line();
  });

  /* ----------------- End Window Resize Actions ----------------- */

  /* --------------------------
    Start Vendors plugins options  
    ----------------------------*/

  /* Start  wow.js  Options */

  /* Start Swiper Options */

  //initialize swiper [Hero Section]
  if ($(".hero-swiper-slider .swiper-container").length) {
    var heroSlider = new Swiper(".hero-swiper-slider .swiper-container", {
      speed: 500,
      spaceBetween: 30,
      loop: true,
      reverseDirection: true,
      // effect: 'fade',
      // fadeEffect: {
      // crossFade: true
      // },
      effect: "coverflow",
      coverflowEffect: {
        stretch: 100,
      },

      autoplay: {
        delay: 800000,
        disableOnInteraction: false,
      },

      navigation: {
        nextEl: ".swiper-button-next",
        prevEl: ".swiper-button-prev",
      },
    });
  }

  // initialize swiper [Testimonials with ONE Column]
  if ($(".testimonials-1-col .swiper-container").length) {
    var testimonialsSlider_1 = new Swiper(
      ".testimonials-1-col .swiper-container",
      {
        // Optional parameters
        speed: 500,
        loop: true,
        grabCursor: true,
        slidesPerView: 1,
        spaceBetween: 50,

        delay: 5000,
        autoplay: {
          delay: 5000,
        },
        breakpoints: {
          991: {
            slidesPerView: 1,
          },
        },

        navigation: {
          nextEl: ".testimonials-1-col .swiper-button-next",
          prevEl: ".testimonials-1-col .swiper-button-prev",
        },
      }
    );
  }

  // initialize swiper [Testimonials with TOW Columns]
  if ($(".testimonials-2-col .swiper-container").length) {
    var testimonialsSlider_2 = new Swiper(
      ".testimonials-2-col .swiper-container",
      {
        // Optional parameters
        speed: 500,
        loop: true,
        grabCursor: true,
        slidesPerView: 2,
        spaceBetween: 20,
        centeredSlides: true,
        delay: 5000,
        autoplay: {
          delay: 5000,
        },
        breakpoints: {
          991: {
            slidesPerView: 1,
          },
        },

        navigation: {
          nextEl: ".testimonials-2-col .swiper-button-next",
          prevEl: ".testimonials-2-col .swiper-button-prev",
        },
      }
    );
  }

  // initialize swiper [Testimonials with THREE Column]
  if ($(".testimonials-3-col .swiper-container").length) {
    var testimonialsSlider_3 = new Swiper(
      ".testimonials-3-col .swiper-container",
      {
        // Optional parameters
        speed: 600,
        loop: true,
        grabCursor: true,
        slidesPerView: 3,
        spaceBetween: 10,
        delay: 5000,
        autoplay: {
          delay: 5000,
        },
        breakpoints: {
          991: {
            slidesPerView: 1,
          },
        },

        navigation: {
          nextEl: ".testimonials-3-col .swiper-button-next",
          prevEl: ".testimonials-3-col .swiper-button-prev",
        },
      }
    );
  }

  //initialize swiper [clients Section]
  if ($(".our-clients .swiper-container").length) {
    var partenersSlider = new Swiper(".our-clients .swiper-container", {
      // Optional parameters
      speed: 600,
      loop: true,
      spaceBetween: 30,
      grabCursor: true,

      delay: 5000,
      autoplay: {
        delay: 5000,
      },
      slidesPerView: 6,
      breakpoints: {
        991: {
          slidesPerView: 3,
          spaceBetween: 20,
        },
      },
    });
  }

  //initialize swiper [single-post page]
  if ($(".post-main-area .post-featured-area .swiper-container").length) {
    var partenersSlider = new Swiper(
      ".post-main-area .post-featured-area .swiper-container",
      {
        // Optional parameters
        slidesPerView: 1,
        grabCursor: true,
        spaceBetween: 0,
        loop: true,
        pagination: {
          el: ".swiper-pagination",
          clickable: true,
        },
        navigation: {
          nextEl: ".swiper-button-next",
          prevEl: ".swiper-button-prev",
        },
      }
    );
  }

  if ($(".portfolio-slider .swiper-container").length) {
    var swiperPortfolioSlider = new Swiper(
      ".portfolio-slider .swiper-container",
      {
        spaceBetween: 10,
        speed: 600,
        loop: true,
        centeredSlides: true,
        slidesPerView: 3,
        autoplay: {
          delay: 5000,
        },
        breakpoints: {
          576: {
            slidesPerView: 1,
            spaceBetween: 10,
          },
          768: {
            slidesPerView: 2,
            spaceBetween: 10,
          },
          991: {
            slidesPerView: 3,
            spaceBetween: 10,
          },
        },

        pagination: {
          el: ".portfolio-slider .swiper-pagination",
          type: "progressbar",
        },

        navigation: {
          nextEl: ".portfolio-slider .swiper-button-next",
          prevEl: ".portfolio-slider .swiper-button-prev",
        },
      }
    );
  }

  /* Start fancybox Options */
  // portfolio fancy box initializer

  if ($("*").fancybox) {
    $().fancybox({
      selector: '[data-fancybox=".filter"]:visible',
      loop: true,
      buttons: ["zoom", "close"],
    });
  }

  /* Start bootstrap Scrollspy Options  */
  /*-------------------------------------*/
  //on one page demos only
  $(bdyOnePage).scrollspy({
    target: navMain,
    offset: navMainHeight + 1,
  });

  /* Start Vegas Slider Options */
  /*-------------------------------------*/

  if (jQuery().vegas) {
    // grab the slides imgs from [data attr in html file]
    var slides = $(".hero-vegas-slider .vegas-slider-content"),
      img_1 = slides.attr("data-vegas-slide-1"),
      img_3 = slides.attr("data-vegas-slide-3"),
      img_2 = slides.attr("data-vegas-slide-2");

    // init vegas slider
    heroVegasSlider.vegas({
      delay: 8000,
      shuffle: false,
      // overlay: '../assets/Images/hero/slider/overlays/04.png',

      animation: [
        "kenburnsUp",
        "kenburnsDown",
        "kenburnsLeft",
        "kenburnsRight",
      ],

      slides: [
        {
          src: img_1, //'../assets/Images/hero/slider/2.jpg'
        },
        {
          src: img_2, //'../assets/Images/hero/slider/1.jpg'
        },
        {
          src: img_3, //'../assets/Images/hero/slider/3.jpg'
        },
      ],
    });
  }

  /* End Vegas counter Options */
  /*-------------------------------------*/

  /* Start isotope Options */
  /*-------------------------------------*/
  if ($(".portfolio .portfolio-btn").length) {
    $(".portfolio .portfolio-btn").on("click", function () {
      $(this).addClass("active").siblings().removeClass("active");

      var $filterValue = $(this).attr("data-filter");
      portfolioGroup.isotope({
        filter: $filterValue,
      });
    });
  }

  var wow = new WOW({
    animateClass: "animated",

    offset: 100,
  });
  wow.init();
  /*----------------- End Vendors plugins options ----------------- */
});

/*----------------- Start page loading Actions -----------------  */

$(window).on("load", function () {
  //Fire the isotope plugin
  if (jQuery().isotope) {
    portfolioGroup.isotope({
      // options
      itemSelector: ".portfolio-item",
      layoutMode: "masonry",
      percentPosition: false,
      filter: "*",
      stagger: 30,
      containerStyle: null,
    });
  }

  //loading screen
  $("#loading-screen").fadeOut(500);
  $("#loading-screen").remove();

  // check if the loaded page have [dat-splitting] attr so the let the page fire splitting.js plugin  ;
  if ($("[data-splitting]").length) {
    Splitting();
  }
});

/*----------------- End Loading Event functions -----------------*/
