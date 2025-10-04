//Defining our animations Header
var timeline = gsap.timeline({defaults: {duration: 1} });

timeline.from(".head",{y : '-100px', ease: 'bounce'})
        .from(".profile",{opacity: 0})
        .from(".nav-item", {opacity: 0,stagger: .5})
        .from("#home-cont",{opacity:0, ease:'power2in'})
        .from(".dynamic",{opacity: 0})
        .from(".texts", {opacity: 0})
        .from(".more",{opacity: 0})
        .from("#butt",{opacity: 0, stagger: .5})
        .from("#imagprofile",{
                x: '150px',
                ease: 'bounce',
                opacity: 0
        })
        .from("#colsocial",{x : '100px', opacity:0, ease: 'power2in'})



//navigation
let next = document.getElementById("next");
let prev = document.getElementById("prev");

next.addEventListener("click", function(){
    let items = document.querySelectorAll(".item");
    document.querySelector(".slide").appendChild(items[0]);
        
});

prev.addEventListener("click", function(){
        let items = document.querySelectorAll(".item");
        document.querySelector(".slide").prepend(items[items.length - 1]);
            
});

//About animation
const scrolltl = gsap.timeline({
        scrollTrigger:{
                trigger: "#About",
                start: 'top bottom',
                end: 'bottom 33%',
                scrub: false,
                toggleActions: "play none none restart"
        }
})

scrolltl.from("#About",{
        opacity: 0,
        duration: 1,
        y: 20
})
        .from("#Aboutcont",{
                x: '-200px',
                opacity: 0,
                duration: 1,
                borderRadius: 0,
        })

        .from(".imageabut",{
                opacity: 0,
                y: 50,
                ease: 'bounce',
                duration: 1
        })
        .from(".infor",{
                color:"gray",
                ease: 'power2',
                y: 30,
                duration: 1,
                stagger: 0.5,
        })

//Service section animation
const servicescroll = gsap.timeline({
        scrollTrigger:{
                trigger: '#Services',
                start: "top bottom",
                end: "bottom center",
                scrub: false,
                toggleActions: "play none none reverse"
        },
        defaults: {
                duration: 1
        }
})

servicescroll.from("#Services", {opacity: 0, y: 20});
servicescroll.from(".servisec", {opacity: 0, stagger: 0.5});

servicescroll.from("#serv",{
        rotate: -40,
        opacity: 0,
        scale: 0.2,
        stagger: 0.5,

})
servicescroll.from(".data",{
        opacity: 0,
        ease: 'bounce',
        y: 30,
        stagger: 0.5,

})

const skilldiv = gsap.timeline({
        scrollTrigger:{
                trigger:".skills",
                start: "top 78%",
                end: "bottom bottom",
                scrub: false,
                toggleActions: "play none none reverse"
        }
});
skilldiv.from(".skills",{
        y: 30,
        ease: 'sine',
        opacity: 0,
        backgroundColor: "black", 
        duration: 1
})
skilldiv.from(".divskills",{
        y:20,
        opacity:0,
        ease: 'elastic',
        stagger: 0.5,
        delay: 0.5,
        duration: 1
        

})

//secion projects
const project = gsap.timeline({
        scrollTrigger:{
                trigger: "#Projects",
                start: "top 90%",
                end: "bottom bottom",
                scrub: false,
                toggleActions: "play none none reverse"
        },
        defaults:{
                duration: 1
        }
        
})
       .from("#Projects",{opacity: 0,y: 20, stagger:0.5})
       .from("#projectcont",{
        opacity: 0,
        stagger: 0.5,
        y:'-4rem',
        ease: 'elastic'
        
        })
        

//contact section
const last = gsap.timeline({
        scrollTrigger:{
                trigger: '#Contact',
                start: "top 90%",
                end: "bottom bottom",
                scrub: false,
                toggleActions: "play none none reverse"

        },
        defaults:{
                duration: 1
        }
})
last.from("#Contact",{opacity: 0, y: 20})
    .from('#formcontainer', {opacity: 0, stagger: 0.5})
    .from(".form_infor",{
        opacity: 0,
        scale: 0.1,
        stagger: 0.5,
        ease:"bounce"
    })
const linlast = gsap.timeline({
        scrollTrigger:{
                trigger:".information",
                start: "top 100%",
                end: "bottom bottom",
                scrub: false,
                toggleActions: "play none none reverse"
        }
})
    .from(".linkss",{
        opacity: 0,
        stagger:{amount: 0.5},
        scale: 0.1
    })