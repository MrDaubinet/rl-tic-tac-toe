<script>
  // Svelte libraries
  import { scale, fade } from 'svelte/transition';
</script>

<div 
  in:scale out:fade
  class="fixed inset-0 flex items-end justify-center px-4 py-6 pointer-events-none sm:p-6 sm:items-start sm:justify-end">
  <!--
    Notification panel, show/hide based on alert state.

    Entering: "transform ease-out duration-300 transition"
      From: "translate-y-2 opacity-0 sm:translate-y-0 sm:translate-x-2"
      To: "translate-y-0 opacity-100 sm:translate-x-0"
    Leaving: "transition ease-in duration-100"
      From: "opacity-100"
      To: "opacity-0"
  -->
  <div class="pr-10 text-right text-white pointer-events-auto">
    <div class="text-xl">
      <slot></slot>
    </div> 
  </div>
</div>